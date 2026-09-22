use crate::tree_store::page_store::fast_hash::{FastHashMapU64, Shrink};
use alloc::collections::VecDeque;
use core::sync::atomic::{AtomicBool, Ordering};

// Every key holds at most one slot in `lru_queue`, for as long as `cache` holds an entry for it.
// A removed key keeps its entry as a tombstone (a `None` value) rather than dropping it, so that
// re-inserting the key revives the tombstone instead of queueing a second slot for it. The
// tombstone is reclaimed when its slot reaches the front of the queue.
//
// Without the tombstone, a page that is freed and later reallocated at the same offset queues a
// fresh slot on every insert, while its earlier slots stay behind. The queue then grows with the
// number of inserts rather than with the number of cached pages, and the cycling in `remove()`
// cannot recover the slots: a key that has been reinserted is live again by the time its stale
// slot reaches the front, so the slot is recycled instead of dropped. A write-heavy workload can
// grow the queue to a million slots against a few thousand cached pages.
#[derive(Default)]
pub struct LRUCache<T> {
    // AtomicBool is the second chance flag. A `None` value is a tombstone: the key has been
    // removed, but its queue slot has not been reclaimed yet.
    cache: FastHashMapU64<(Option<T>, AtomicBool)>,
    lru_queue: VecDeque<u64>,
    // Number of entries whose value is `Some`. `cache.len()` also counts tombstones.
    live: usize,
}

impl<T> LRUCache<T> {
    pub(crate) fn new() -> Self {
        Self {
            cache: FastHashMapU64::default(),
            lru_queue: VecDeque::default(),
            live: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.live
    }

    pub(crate) fn insert(&mut self, key: u64, value: T) -> Option<T> {
        if let Some((slot, second_chance)) = self.cache.get_mut(&key) {
            second_chance.store(false, Ordering::Release);
            let previous = slot.replace(value);
            if previous.is_none() {
                // Revived a tombstone. Its queue slot is still in place, so none is pushed.
                self.live += 1;
            }
            previous
        } else {
            self.cache
                .insert(key, (Some(value), AtomicBool::new(false)));
            self.lru_queue.push_back(key);
            self.live += 1;
            None
        }
    }

    pub(crate) fn remove(&mut self, key: u64) -> Option<T> {
        let value = self.cache.get_mut(&key).and_then(|(slot, _)| slot.take());
        if value.is_some() {
            self.live -= 1;
            // Reclaim a couple of tombstones, so that a cache which is only ever removed from
            // does not hold their entries until the next eviction. Live slots are recycled to the
            // back of the queue, as they were before.
            if self.lru_queue.len() > 2 * self.live {
                for _ in 0..2 {
                    let Some(front) = self.lru_queue.pop_front() else {
                        break;
                    };
                    match self.cache.get(&front) {
                        Some((Some(_), second_chance)) => {
                            second_chance.store(false, Ordering::Release);
                            self.lru_queue.push_back(front);
                        }
                        Some((None, _)) => {
                            self.cache.remove(&front);
                        }
                        None => {}
                    }
                }
            }
        }
        value
    }

    pub(crate) fn get(&self, key: u64) -> Option<&T> {
        if let Some((Some(value), second_chance)) = self.cache.get(&key) {
            second_chance.store(true, Ordering::Release);
            Some(value)
        } else {
            None
        }
    }

    pub(crate) fn get_mut(&mut self, key: u64) -> Option<&mut T> {
        if let Some((Some(value), second_chance)) = self.cache.get_mut(&key) {
            second_chance.store(true, Ordering::Release);
            Some(value)
        } else {
            None
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&u64, &T)> {
        self.cache
            .iter()
            .filter_map(|(key, (value, _))| value.as_ref().map(|value| (key, value)))
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (&u64, &mut T)> {
        self.cache
            .iter_mut()
            .filter_map(|(key, (value, _))| value.as_mut().map(|value| (key, value)))
    }

    pub(crate) fn pop_lowest_priority(&mut self) -> Option<(u64, T)> {
        while let Some(key) = self.lru_queue.pop_front() {
            match self.cache.get(&key) {
                Some((Some(_), second_chance)) => {
                    if second_chance
                        .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        self.lru_queue.push_back(key);
                    } else {
                        let (value, _) = self.cache.remove(&key).unwrap();
                        self.live -= 1;
                        return Some((key, value.unwrap()));
                    }
                }
                // A tombstone reaching the front is what reclaims its entry.
                Some((None, _)) => {
                    self.cache.remove(&key);
                }
                None => {}
            }
        }
        None
    }

    pub(crate) fn clear(&mut self) {
        self.cache.shrink();
        self.cache.clear();
        self.lru_queue.shrink_to_fit();
        self.lru_queue.clear();
        self.live = 0;
    }
}

#[cfg(test)]
mod test {
    use super::LRUCache;

    #[test]
    fn reinserting_a_removed_key_does_not_grow_the_queue() {
        let mut cache: LRUCache<u64> = LRUCache::new();
        for key in 0..16u64 {
            cache.insert(key, key);
        }
        // Churn the same keys the way a page that is freed and reallocated does.
        for _ in 0..10_000 {
            for key in 0..16u64 {
                assert_eq!(cache.remove(key), Some(key));
                assert!(cache.insert(key, key).is_none());
            }
        }
        assert_eq!(cache.len(), 16);
        assert!(
            cache.lru_queue.len() <= 32,
            "queue grew to {} slots for 16 keys",
            cache.lru_queue.len()
        );
    }

    #[test]
    fn eviction_returns_every_live_entry_once() {
        let mut cache: LRUCache<u64> = LRUCache::new();
        for key in 0..64u64 {
            cache.insert(key, key);
        }
        for key in 0..64u64 {
            if key % 2 == 0 {
                assert_eq!(cache.remove(key), Some(key));
            }
        }
        let mut evicted = alloc::vec::Vec::new();
        while let Some((key, value)) = cache.pop_lowest_priority() {
            assert_eq!(key, value);
            evicted.push(key);
        }
        evicted.sort_unstable();
        let expected: alloc::vec::Vec<u64> = (0..64u64).filter(|key| key % 2 == 1).collect();
        assert_eq!(evicted, expected);
        assert_eq!(cache.len(), 0);
        assert!(cache.lru_queue.is_empty());
        assert!(cache.cache.is_empty());
    }

    #[test]
    fn a_second_chance_defers_eviction() {
        let mut cache: LRUCache<u64> = LRUCache::new();
        cache.insert(1, 1);
        cache.insert(2, 2);
        // Touching key 1 sets its second chance flag, so key 2 is evicted first.
        assert_eq!(cache.get(1), Some(&1));
        assert_eq!(cache.pop_lowest_priority(), Some((2, 2)));
        assert_eq!(cache.pop_lowest_priority(), Some((1, 1)));
        assert_eq!(cache.pop_lowest_priority(), None);
    }
}
