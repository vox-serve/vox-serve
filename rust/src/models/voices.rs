use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};

/// A reference voice encoded into one of the program's voice slots. Requests hold an `Arc` until they
/// finish (`Prompt::hold`); a slot is only re-encoded once no request holds its voice.
pub struct Voice {
    pub slot: i32,
}

/// Per slot: (content hash, voice, last use), and the use clock.
type Slots = (Vec<Option<(u64, Arc<Voice>, u64)>>, u64);

/// Voice slots: 0 holds the default voice, the others an LRU of reference clips keyed by content
/// (`vox_serve.rt.prompt.VoiceSlots` on the Python side).
pub struct VoiceSlots {
    slots: Mutex<Slots>,
    /// Serializes encodes (the Python hooks run one at a time anyway), so one clip is encoded once.
    encode: Mutex<()>,
}

impl VoiceSlots {
    pub fn new(n: usize) -> Self {
        let mut slots = vec![None; n.max(1)];
        slots[0] = Some((0, Arc::new(Voice { slot: 0 }), 0));
        Self { slots: Mutex::new((slots, 0)), encode: Mutex::new(()) }
    }

    pub fn default_voice(&self) -> Arc<Voice> {
        self.slots.lock().unwrap().0[0].as_ref().unwrap().1.clone()
    }

    fn lookup(&self, key: u64) -> Option<Arc<Voice>> {
        let mut g = self.slots.lock().unwrap();
        g.1 += 1;
        let now = g.1;
        g.0.iter_mut().skip(1).flatten().find(|(k, _, _)| *k == key).map(|e| {
            e.2 = now;
            e.1.clone()
        })
    }

    /// An empty slot, or the least recently used one no request holds (its entry is dropped).
    fn reserve(&self) -> Option<usize> {
        let mut g = self.slots.lock().unwrap();
        let slots = &mut g.0;
        if let Some(i) = (1..slots.len()).find(|&i| slots[i].is_none()) {
            return Some(i);
        }
        let i = (1..slots.len())
            .filter(|&i| slots[i].as_ref().is_some_and(|(_, v, _)| Arc::strong_count(v) == 1))
            .min_by_key(|&i| slots[i].as_ref().unwrap().2)?;
        slots[i] = None;
        Some(i)
    }

    /// The voice of reference clip `path`; on a miss, `encode(slot)` writes it into a free slot, waiting
    /// (up to 120 s) for running requests to release one.
    pub fn get(&self, path: &str, encode: impl FnOnce(usize) -> Result<()>) -> Result<Arc<Voice>> {
        let bytes = std::fs::read(path).map_err(|e| anyhow!("reading reference audio: {e}"))?;
        let mut h = DefaultHasher::new();
        bytes.hash(&mut h);
        let key = h.finish();
        if let Some(v) = self.lookup(key) {
            return Ok(v);
        }
        let n = self.slots.lock().unwrap().0.len();
        if n < 2 {
            bail!("reference audio needs a voice slot (n_voices >= 2)");
        }
        let _encoding = self.encode.lock().unwrap();
        let deadline = Instant::now() + Duration::from_secs(120);
        let slot = loop {
            if let Some(v) = self.lookup(key) {
                return Ok(v);
            }
            if let Some(i) = self.reserve() {
                break i;
            }
            if Instant::now() >= deadline {
                bail!("all {} voice slots are in use by running requests", n - 1);
            }
            // Running requests release their voices when they finish.
            std::thread::sleep(Duration::from_millis(20));
        };
        encode(slot)?;
        let v = Arc::new(Voice { slot: slot as i32 });
        let mut g = self.slots.lock().unwrap();
        let now = g.1;
        g.0[slot] = Some((key, v.clone(), now));
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::VoiceSlots;

    #[test]
    fn reuses_and_evicts_unheld_voices() {
        let dir = std::env::temp_dir().join(format!("vox_voices_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let clip = |name: &str| {
            let p = dir.join(name);
            std::fs::write(&p, name.as_bytes()).unwrap();
            p.to_string_lossy().to_string()
        };
        let (a, b, c) = (clip("a"), clip("b"), clip("c"));
        let slots = VoiceSlots::new(3);
        let encodes = Cell::new(0);
        let get = |p: &str| {
            slots
                .get(p, |_| {
                    encodes.set(encodes.get() + 1);
                    Ok(())
                })
                .unwrap()
        };
        assert_eq!(slots.default_voice().slot, 0);
        let va = get(&a);
        let vb = get(&b);
        assert_eq!((va.slot, vb.slot), (1, 2));
        assert_eq!(get(&a).slot, 1); // same content: no new encode
        assert_eq!(encodes.get(), 2);
        drop(vb); // b is no longer held: c takes its slot, a (held) keeps its own
        assert_eq!(get(&c).slot, 2);
        assert_eq!(get(&a).slot, 1);
        assert_eq!(encodes.get(), 3);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
