//! Host-side allocator for the program's device feature pool (`LMProgram.feature_pool`).
//!
//! A request's prompt features (e.g. encoded input audio) occupy a contiguous row range from
//! preprocessing until its prefill has run; the lease returns the range when dropped.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use anyhow::{bail, Result};

#[derive(Debug)]
struct Inner {
    /// Free ranges `(start, len)`, sorted by start and coalesced.
    free: Mutex<Vec<(usize, usize)>>,
    freed: Condvar,
    rows: usize,
}

#[derive(Debug, Clone)]
pub struct FeaturePool(Arc<Inner>);

/// A contiguous range of pool rows, freed on drop.
#[derive(Debug)]
pub struct FeatureLease {
    pool: Arc<Inner>,
    pub offset: usize,
    pub len: usize,
}

impl FeaturePool {
    pub fn new(rows: usize) -> Self {
        let free = if rows > 0 { vec![(0, rows)] } else { Vec::new() };
        Self(Arc::new(Inner { free: Mutex::new(free), freed: Condvar::new(), rows }))
    }

    /// First-fit allocation, waiting up to `timeout` for other requests to release rows.
    pub fn alloc(&self, len: usize, timeout: Duration) -> Result<FeatureLease> {
        if len == 0 || len > self.0.rows {
            bail!("feature rows {len} out of range (pool has {})", self.0.rows);
        }
        let mut free = self.0.free.lock().unwrap();
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(i) = free.iter().position(|&(_, n)| n >= len) {
                let (start, n) = free[i];
                if n == len {
                    free.remove(i);
                } else {
                    free[i] = (start + len, n - len);
                }
                return Ok(FeatureLease { pool: self.0.clone(), offset: start, len });
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                bail!("feature pool exhausted ({len} rows requested)");
            }
            free = self.0.freed.wait_timeout(free, deadline - now).unwrap().0;
        }
    }
}

impl Drop for FeatureLease {
    fn drop(&mut self) {
        let mut free = self.pool.free.lock().unwrap();
        let i = free.partition_point(|&(s, _)| s < self.offset);
        free.insert(i, (self.offset, self.len));
        if i + 1 < free.len() && free[i].0 + free[i].1 == free[i + 1].0 {
            free[i].1 += free[i + 1].1;
            free.remove(i + 1);
        }
        if i > 0 && free[i - 1].0 + free[i - 1].1 == free[i].0 {
            free[i - 1].1 += free[i].1;
            free.remove(i);
        }
        self.pool.freed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_coalesce() {
        let p = FeaturePool::new(10);
        let t = Duration::from_millis(1);
        let a = p.alloc(4, t).unwrap();
        let b = p.alloc(4, t).unwrap();
        assert!(p.alloc(4, t).is_err());
        assert_eq!((a.offset, b.offset), (0, 4));
        drop(a);
        drop(b);
        let c = p.alloc(10, t).unwrap();
        assert_eq!(c.offset, 0);
    }
}
