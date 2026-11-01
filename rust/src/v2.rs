//! Knobs of the v2 (asynchronous, SM-partitioned) engine loop.
//!
//! The LM and detok lanes each keep one batch in flight and are collected by their completion
//! events. Which SM partition (mode) the next LM step and detok batch use, whether the detok
//! batch waits for the LM's depth phase, and how many ready chunks the detok batch waits for are
//! all policy decisions made here, so experiments can vary them from the command line.

use anyhow::{bail, Result};

/// How the GPU is partitioned between the two lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModePolicy {
    /// Always use SM mode `m` for decode steps and detok batches (0 = both lanes on all SMs).
    Fixed(usize),
    /// Mode `mode` while at least `min_rows` requests are decoding, otherwise mode 0.
    Auto { min_rows: usize, mode: usize },
    /// Pick the mode with the lowest predicted time per LM step from latencies measured online
    /// (see `Calibration`), exploring modes that have no measurement yet.
    Adaptive,
}

impl std::str::FromStr for ModePolicy {
    type Err = anyhow::Error;
    /// `<m>` or `auto:<min_rows>:<m>`.
    fn from_str(s: &str) -> Result<Self> {
        if s == "adaptive" {
            return Ok(Self::Adaptive);
        }
        if let Some(rest) = s.strip_prefix("auto:") {
            let p: Vec<&str> = rest.split(':').collect();
            if p.len() != 2 {
                bail!("expected auto:<min_rows>:<mode>, got {s}");
            }
            return Ok(Self::Auto { min_rows: p[0].parse()?, mode: p[1].parse()? });
        }
        Ok(Self::Fixed(s.parse()?))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateMode {
    /// Detok batches start whenever they are ready.
    None,
    /// Detok batches start only once the LM's current step has entered its depth graph
    /// (split LM steps only; otherwise behaves like `none`).
    Depth,
    /// Detok batches start once the LM's current step has reached layer group `i` (layer-split steps).
    Layer(usize),
}

impl std::str::FromStr for GateMode {
    type Err = anyhow::Error;
    /// `none`, `depth` or `layer:<group>`.
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "none" => Ok(Self::None),
            "depth" => Ok(Self::Depth),
            _ => match s.strip_prefix("layer:") {
                Some(i) => Ok(Self::Layer(i.parse()?)),
                None => bail!("expected none, depth or layer:<group>, got {s}"),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct V2Config {
    pub enabled: bool,
    /// Ready continuation chunks a detok batch waits for (0 = decode as soon as anything is ready).
    pub dec_batch: usize,
    /// Largest non-urgent detok batch (0 = the largest captured size). Bounds how long an urgent
    /// chunk can wait behind a background batch, which cannot be interrupted once launched.
    pub dec_max_batch: usize,
    /// Flush a partial batch once the earliest playback deadline minus the expected detok time is
    /// within this margin (seconds).
    pub dec_flush: f64,
    /// A chunk whose playback deadline is within this margin (seconds) is urgent, like a first chunk.
    pub urgent: f64,
    pub gate: GateMode,
    pub mode: ModePolicy,
    /// Prefill steps wait for an in-flight detok batch and block new ones until they finish.
    pub prefill_exclusive: bool,
    /// Batches with urgent chunks run on all SMs (mode 0) while the LM lane waits.
    pub urgent_exclusive: bool,
    /// Urgency from playback deadlines only: a first chunk is due `--deadline-startup-ms` after
    /// arrival instead of always being urgent, and the urgency margin includes the measured latency
    /// of the largest detok batch (first audio as a threshold, not as short as possible).
    pub slack_urgency: bool,
    /// Strict time sharing: every LM step and detok batch waits (on the GPU) for the other lane, so
    /// they alternate on the whole GPU; detok batching (`dec_batch`) still applies. The "batching
    /// without colocation" point of comparison.
    pub time_share: bool,
}

impl Default for V2Config {
    fn default() -> Self {
        Self {
            enabled: false,
            dec_batch: 0,
            dec_max_batch: 0,
            dec_flush: 0.05,
            urgent: 0.1,
            gate: GateMode::None,
            mode: ModePolicy::Fixed(0),
            prefill_exclusive: false,
            urgent_exclusive: false,
            slack_urgency: false,
            time_share: false,
        }
    }
}

impl V2Config {
    /// SM mode for the next decode step, given the decoding rows and the number of modes.
    pub fn mode_for(&self, decode_rows: usize, n_modes: usize) -> usize {
        let m = match self.mode {
            ModePolicy::Fixed(m) => m,
            ModePolicy::Auto { min_rows, mode } => {
                if decode_rows >= min_rows {
                    mode
                } else {
                    0
                }
            }
            ModePolicy::Adaptive => 0,
        };
        m.min(n_modes.saturating_sub(1))
    }

    /// SM mode for a background detok batch: the split the policy gives decode steps (the adaptive
    /// policy: that of the latest decode step), never the whole-GPU mode of a prefill step.
    pub fn background_mode(&self, decode_rows: usize, last_decode_mode: usize, n_modes: usize) -> usize {
        match self.mode {
            ModePolicy::Adaptive => last_decode_mode.min(n_modes.saturating_sub(1)),
            _ => self.mode_for(decode_rows, n_modes),
        }
    }
}

/// Online latency table for the adaptive mode policy: exponentially weighted host-observed latency of
/// LM steps and detok batches per (mode, captured size). Latencies are measured while the other
/// lane runs as usual, so they include interference.
#[derive(Debug, Default)]
pub struct Calibration {
    lm: Vec<(usize, usize, f64, u32)>,
    dt: Vec<(usize, usize, f64, u32)>,
    since_explore: u32,
}

impl Calibration {
    fn update(t: &mut Vec<(usize, usize, f64, u32)>, mode: usize, size: usize, secs: f64) {
        match t.iter_mut().find(|e| e.0 == mode && e.1 == size) {
            Some(e) => {
                e.2 = 0.8 * e.2 + 0.2 * secs;
                e.3 += 1;
            }
            None => t.push((mode, size, secs, 1)),
        }
    }
    /// Latency at `size` in `mode`: the measurement at the nearest size with at least three samples,
    /// scaled linearly by size when `per_item` (detok batches), taken as is otherwise (LM steps,
    /// whose cost is nearly flat in batch size).
    fn get(t: &[(usize, usize, f64, u32)], mode: usize, size: usize, per_item: bool) -> Option<f64> {
        t.iter().filter(|e| e.0 == mode && e.3 >= 3).min_by_key(|e| e.1.abs_diff(size)).map(|e| {
            if per_item {
                e.2 * size as f64 / e.1 as f64
            } else {
                e.2
            }
        })
    }
    pub fn lm_step(&mut self, mode: usize, bucket: usize, secs: f64) {
        Self::update(&mut self.lm, mode, bucket, secs);
    }
    pub fn detok(&mut self, mode: usize, bucket: usize, secs: f64) {
        Self::update(&mut self.dt, mode, bucket, secs);
    }

    /// Mode for a decode step of `bucket` rows producing `chunks_per_step` detok chunks, decoded in
    /// batches of `dt_bucket`. Predicted time per step = max(LM step, detok work per step), since the
    /// two lanes run concurrently. Modes without three measurements yet are tried first, at most
    /// once every 8 steps.
    pub fn choose(&mut self, n_modes: usize, bucket: usize, chunks_per_step: f64, dt_bucket: usize) -> usize {
        self.since_explore += 1;
        let mut best = (f64::INFINITY, 0);
        for m in 0..n_modes {
            let (Some(lm), Some(dt)) = (Self::get(&self.lm, m, bucket, false), Self::get(&self.dt, m, dt_bucket, true))
            else {
                if self.since_explore >= 8 {
                    self.since_explore = 0;
                    return m;
                }
                continue;
            };
            let per_step = lm.max(chunks_per_step * dt / dt_bucket as f64);
            if per_step < best.0 {
                best = (per_step, m);
            }
        }
        best.1
    }
}

/// How many of `n_ready` candidate chunks (sorted urgent-first, then by deadline) a detok batch
/// takes now. `buckets` are the captured detok batch sizes.
pub fn detok_take(
    n_ready: usize,
    n_urgent: usize,
    target: usize,
    cap: usize,
    lm_active: bool,
    deadline_due: bool,
    buckets: &[usize],
) -> usize {
    if n_ready == 0 {
        return 0;
    }
    let fill = |n: usize| -> usize {
        // Round up to the graph bucket that `n` lands in: the padding rows are free.
        let b = buckets.iter().copied().find(|&b| b >= n).unwrap_or(cap);
        b.min(n_ready).min(cap)
    };
    if n_urgent > 0 {
        return fill(n_urgent.max(target.min(n_ready)));
    }
    if target == 0 {
        return n_ready.min(cap);
    }
    if n_ready >= target {
        // The target is a minimum: take everything that is ready (larger batches are cheaper per chunk).
        return n_ready.min(cap);
    }
    if !lm_active || deadline_due {
        return n_ready.min(cap);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_prefers_faster_mode() {
        let mut c = Calibration::default();
        for _ in 0..3 {
            // Mode 0: fast LM, slow detok; mode 1: slightly slower LM, detok hidden.
            c.lm_step(0, 32, 0.010);
            c.detok(0, 8, 0.040);
            c.lm_step(1, 32, 0.011);
            c.detok(1, 4, 0.010);
        }
        // 3.2 chunks per step: mode 0 = max(10, 3.2 * 40 / 8 = 16) ms, mode 1 = max(11, 3.2 * 20 / 8 = 8) ms.
        assert_eq!(c.choose(2, 32, 3.2, 8), 1);
        // With few chunks per step the faster LM wins.
        assert_eq!(c.choose(2, 32, 0.5, 8), 0);
    }

    #[test]
    fn background_mode_ignores_prefill() {
        let mut cfg = V2Config { mode: ModePolicy::Fixed(1), ..Default::default() };
        // The latest LM step was a prefill (mode 0): background batches still use the split.
        assert_eq!(cfg.background_mode(0, 0, 4), 1);
        assert_eq!(cfg.background_mode(40, 0, 4), 1);
        assert_eq!(cfg.background_mode(40, 0, 1), 0);
        cfg.mode = ModePolicy::Auto { min_rows: 16, mode: 2 };
        assert_eq!(cfg.background_mode(8, 2, 4), 0);
        assert_eq!(cfg.background_mode(16, 0, 4), 2);
        cfg.mode = ModePolicy::Adaptive;
        assert_eq!(cfg.background_mode(16, 3, 4), 3);
        assert_eq!(cfg.background_mode(16, 3, 2), 1);
    }

    #[test]
    fn parses_mode_policies() {
        assert_eq!("2".parse::<ModePolicy>().unwrap(), ModePolicy::Fixed(2));
        assert_eq!("auto:16:1".parse::<ModePolicy>().unwrap(), ModePolicy::Auto { min_rows: 16, mode: 1 });
        assert!("auto:1".parse::<ModePolicy>().is_err());
    }

    #[test]
    fn detok_batching() {
        let b = [1, 2, 4, 8, 16];
        // As soon as ready.
        assert_eq!(detok_take(3, 0, 0, 16, true, false, &b), 3);
        // Waits for the target while the LM keeps producing chunks.
        assert_eq!(detok_take(3, 0, 8, 16, true, false, &b), 0);
        assert_eq!(detok_take(9, 0, 8, 16, true, false, &b), 9);
        assert_eq!(detok_take(20, 0, 8, 16, true, false, &b), 16);
        // Flushes when a deadline is due or the LM has nothing left to produce.
        assert_eq!(detok_take(3, 0, 8, 16, true, true, &b), 3);
        assert_eq!(detok_take(3, 0, 8, 16, false, false, &b), 3);
        // Urgent chunks go now, padded up to their bucket with other ready chunks.
        assert_eq!(detok_take(6, 3, 8, 16, true, false, &b), 6);
        assert_eq!(detok_take(10, 3, 0, 16, true, false, &b), 4);
    }
}
