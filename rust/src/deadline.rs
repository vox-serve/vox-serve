//! GPU-independent policy math. Costs are configurable estimates in seconds.
//! A deadline is relative to now; negative slack means that the request is late.

#[derive(Debug, Clone)]
pub struct DeadlineConfig {
    pub startup: f64,
    pub lm_step: f64,
    pub codec: f64,
    pub prefill: f64,
    pub max_prefill_deferral: f64,
    /// Zero disables backpressure. Counts complete chunks, including in-flight work.
    pub max_pending_chunks: usize,
}

impl DeadlineConfig {
    pub fn valid(&self) -> bool {
        [self.startup, self.lm_step, self.codec, self.prefill, self.max_prefill_deferral]
            .iter()
            .all(|x| x.is_finite() && *x > 0.0)
    }
}

pub fn ready_chunks(frames: usize, start: usize, interval: usize, step: usize) -> usize {
    if frames.saturating_sub(start) < interval {
        0
    } else {
        1 + (frames - start - interval) / step
    }
}

/// Slack for producing the first chunk whose tokens are not complete yet. Already
/// queued audio buys playback time; it must not make an ahead-of-playback LM urgent.
pub fn lm_slack(deadline: f64, queued: usize, chunk_secs: f64, remaining: usize, cfg: &DeadlineConfig) -> f64 {
    deadline + queued as f64 * chunk_secs - remaining as f64 * cfg.lm_step - cfg.codec
}

pub fn defer_prefill(min_slack: f64, deferred_for: f64, cfg: &DeadlineConfig) -> bool {
    min_slack < cfg.prefill && deferred_for < cfg.max_prefill_deferral
}

pub fn throttled(pending: usize, cfg: &DeadlineConfig) -> bool {
    cfg.max_pending_chunks > 0 && pending >= cfg.max_pending_chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> DeadlineConfig {
        DeadlineConfig {
            startup: 0.5,
            lm_step: 0.01,
            codec: 0.03,
            prefill: 0.03,
            max_prefill_deferral: 0.1,
            max_pending_chunks: 2,
        }
    }

    #[test]
    fn distinguishes_urgent_streams_and_accounts_for_queued_audio() {
        let cfg = config();
        assert!(lm_slack(0.15, 0, 0.8, 10, &cfg) < lm_slack(0.5, 0, 0.8, 10, &cfg));
        assert!(lm_slack(0.15, 1, 0.8, 10, &cfg) > lm_slack(0.5, 0, 0.8, 10, &cfg));
        assert!(lm_slack(-0.1, 0, 0.8, 10, &cfg) < 0.0);
    }

    #[test]
    fn handles_partial_chunks_and_overlapping_codecs() {
        assert_eq!(ready_chunks(9, 0, 10, 10), 0);
        assert_eq!(ready_chunks(20, 0, 10, 10), 2);
        assert_eq!(ready_chunks(20, 10, 10, 10), 1);
        assert_eq!(ready_chunks(14, 0, 10, 8), 1);
        assert_eq!(ready_chunks(18, 0, 10, 8), 2);
        assert_eq!(ready_chunks(3, 10, 10, 10), 0);
    }

    #[test]
    fn admission_protects_decode_but_has_bounded_deferral() {
        let cfg = config();
        assert!(defer_prefill(-1.0, 0.0, &cfg));
        assert!(!defer_prefill(-1.0, 0.1, &cfg));
        assert!(!defer_prefill(0.1, 0.0, &cfg));
    }

    #[test]
    fn backpressure_is_optional_and_releases_after_codec_progress() {
        let mut cfg = config();
        assert!(throttled(2, &cfg));
        assert!(!throttled(1, &cfg));
        cfg.max_pending_chunks = 0;
        assert!(!throttled(100, &cfg));
        cfg.lm_step = f64::NAN;
        assert!(!cfg.valid());
    }
}
