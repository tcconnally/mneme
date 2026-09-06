//! #930: deterministic recall-eval regression detection.
//!
//! Pure threshold/regression computation for scheduled evaluation runs.
//! No IO, no LLM: given the current run's metric rates and the trailing
//! history of prior runs (same suite + eval kind), decide which metrics
//! breached their floor/cap or regressed past a delta against the trailing
//! mean. Callers own windowing (db.rs truncates to the trailing window).
//!
//! Direction is per-metric: most rates are higher-is-better, but
//! `scope_invalid_recall_rate` and `stale_recall_rate` are lower-is-better
//! (see benchmark/quality README). A metric with no baseline is
//! floor-checked only. Metrics absent from the current run, with
//! `status != "available"`-style markers, or with non-finite rates are
//! skipped by the caller before reaching this module.

/// Direction of a metric's goodness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    HigherIsBetter,
    LowerIsBetter,
}

impl Direction {
    pub fn label(&self) -> &'static str {
        match self {
            Direction::HigherIsBetter => "higher_better",
            Direction::LowerIsBetter => "lower_better",
        }
    }
}

/// Metrics where a HIGHER rate is worse (rates that must stay low).
pub const LOWER_IS_BETTER: &[&str] = &["scope_invalid_recall_rate", "stale_recall_rate"];

/// Per-metric thresholds. `floor` is an absolute floor for higher-is-better
/// metrics and an absolute cap for lower-is-better ones. `regression_delta`
/// is the minimum absolute change against the trailing mean that counts as a
/// regression (in the bad direction). A value of `0` disables the check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalThresholds {
    pub floor: f64,
    pub regression_delta: f64,
}

impl Default for EvalThresholds {
    fn default() -> Self {
        EvalThresholds {
            floor: 0.90,
            regression_delta: 0.05,
        }
    }
}

/// One detected breach.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Breach {
    pub metric: String,
    pub current: f64,
    /// Trailing mean over prior runs; None when there is no baseline.
    pub trailing_mean: Option<f64>,
    /// current - trailing_mean; None when there is no baseline.
    pub delta: Option<f64>,
    /// "floor" (absolute floor/cap) or "regression" (delta vs trailing mean).
    pub threshold_type: String,
    /// "higher_better" or "lower_better".
    pub direction: String,
}

/// Direction for a metric name; defaults to higher-is-better.
pub fn direction_for(metric: &str) -> Direction {
    if LOWER_IS_BETTER.contains(&metric) {
        Direction::LowerIsBetter
    } else {
        Direction::HigherIsBetter
    }
}

/// Compute breaches for `current` given `history` (prior runs, oldest first)
/// and per-metric `thresholds`. A metric missing from `thresholds` uses
/// defaults. `history` entries are rate maps from prior runs; only metrics
/// present in the current run are evaluated.
pub fn compute_regression(
    current: &std::collections::BTreeMap<String, f64>,
    history: &[std::collections::BTreeMap<String, f64>],
    thresholds: &std::collections::BTreeMap<String, EvalThresholds>,
) -> Vec<Breach> {
    let mut breaches = Vec::new();
    for (metric, &current_rate) in current {
        if !current_rate.is_finite() {
            continue; // non-finite rates carry no signal
        }
        let direction = direction_for(metric);
        let t = thresholds.get(metric).copied().unwrap_or_default();
        // Absolute floor/cap check (independent of any baseline).
        if t.floor > 0.0 {
            let breached = match direction {
                Direction::HigherIsBetter => current_rate < t.floor,
                Direction::LowerIsBetter => current_rate > t.floor,
            };
            if breached {
                breaches.push(Breach {
                    metric: metric.clone(),
                    current: current_rate,
                    trailing_mean: None,
                    delta: None,
                    threshold_type: "floor".to_string(),
                    direction: direction.label().to_string(),
                });
            }
        }
        // Regression check against the trailing mean (needs a baseline).
        if t.regression_delta > 0.0 && !history.is_empty() {
            let mut sum = 0.0_f64;
            let mut n = 0_usize;
            for prior in history {
                if let Some(&v) = prior.get(metric) {
                    if v.is_finite() {
                        sum += v;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                let mean = sum / n as f64;
                let delta = current_rate - mean;
                let regressed = match direction {
                    Direction::HigherIsBetter => delta <= -t.regression_delta,
                    Direction::LowerIsBetter => delta >= t.regression_delta,
                };
                if regressed {
                    breaches.push(Breach {
                        metric: metric.clone(),
                        current: current_rate,
                        trailing_mean: Some(mean),
                        delta: Some(delta),
                        threshold_type: "regression".to_string(),
                        direction: direction.label().to_string(),
                    });
                }
            }
        }
    }
    breaches
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn rates(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn thresh(floor: f64, regression_delta: f64) -> BTreeMap<String, EvalThresholds> {
        let mut m = BTreeMap::new();
        m.insert(
            "validity_rate".to_string(),
            EvalThresholds {
                floor,
                regression_delta,
            },
        );
        m.insert(
            "scope_invalid_recall_rate".to_string(),
            EvalThresholds {
                floor,
                regression_delta,
            },
        );
        m
    }

    #[test]
    fn stable_high_metric_has_no_breach() {
        let current = rates(&[("validity_rate", 1.0)]);
        let history = vec![
            rates(&[("validity_rate", 0.98)]),
            rates(&[("validity_rate", 0.99)]),
        ];
        let breaches = compute_regression(&current, &history, &thresh(0.9, 0.05));
        assert!(breaches.is_empty(), "expected no breach, got {breaches:?}");
    }

    #[test]
    fn floor_breach_fires_without_any_baseline() {
        let current = rates(&[("validity_rate", 0.85)]);
        let breaches = compute_regression(&current, &[], &thresh(0.9, 0.05));
        assert_eq!(breaches.len(), 1);
        assert_eq!(breaches[0].threshold_type, "floor");
        assert_eq!(breaches[0].direction, "higher_better");
        assert!(breaches[0].trailing_mean.is_none());
        assert_eq!(breaches[0].current, 0.85);
    }

    #[test]
    fn regression_breach_when_drop_exceeds_delta() {
        let current = rates(&[("validity_rate", 0.90)]);
        let history = vec![
            rates(&[("validity_rate", 0.98)]),
            rates(&[("validity_rate", 0.98)]),
        ];
        let breaches = compute_regression(&current, &history, &thresh(0.9, 0.05));
        assert_eq!(breaches.len(), 1);
        assert_eq!(breaches[0].threshold_type, "regression");
        assert_eq!(breaches[0].trailing_mean, Some(0.98));
        assert!((breaches[0].delta.unwrap() - (-0.08)).abs() < 1e-9);
    }

    #[test]
    fn no_regression_breach_within_delta() {
        let current = rates(&[("validity_rate", 0.95)]);
        let history = vec![rates(&[("validity_rate", 0.98)])];
        let breaches = compute_regression(&current, &history, &thresh(0.9, 0.05));
        assert!(breaches.is_empty(), "expected no breach, got {breaches:?}");
    }

    #[test]
    fn lower_is_better_cap_and_increase_breach() {
        // scope_invalid_recall_rate is lower-is-better: cap 0.10, regression
        // when current >= trailing_mean + 0.05.
        let current = rates(&[("scope_invalid_recall_rate", 0.20)]);
        let history = vec![rates(&[("scope_invalid_recall_rate", 0.05)])];
        let breaches = compute_regression(&current, &history, &thresh(0.1, 0.05));
        assert_eq!(
            breaches.len(),
            2,
            "cap and regression both breached: {breaches:?}"
        );
        assert!(breaches.iter().all(|b| b.direction == "lower_better"));
        assert!(breaches.iter().any(|b| b.threshold_type == "floor"));
        assert!(breaches.iter().any(|b| b.threshold_type == "regression"));
    }

    #[test]
    fn lower_is_better_improvement_is_not_a_breach() {
        let current = rates(&[("scope_invalid_recall_rate", 0.0)]);
        let history = vec![rates(&[("scope_invalid_recall_rate", 0.05)])];
        let breaches = compute_regression(&current, &history, &thresh(0.1, 0.05));
        assert!(breaches.is_empty(), "expected no breach, got {breaches:?}");
    }

    #[test]
    fn zero_thresholds_disable_checks() {
        let current = rates(&[("validity_rate", 0.85)]);
        let history = vec![rates(&[("validity_rate", 0.98)])];
        // floor=0 disables the floor; regression_delta=0 disables delta checks.
        let breaches = compute_regression(&current, &history, &thresh(0.0, 0.0));
        assert!(breaches.is_empty(), "expected no breach, got {breaches:?}");
    }

    #[test]
    fn missing_current_metric_is_skipped() {
        let current = rates(&[("validity_rate", 1.0)]);
        let history = vec![rates(&[("scope_invalid_recall_rate", 0.5)])];
        let breaches = compute_regression(&current, &history, &thresh(0.9, 0.05));
        assert!(breaches.is_empty(), "expected no breach, got {breaches:?}");
    }

    #[test]
    fn non_finite_current_rate_is_skipped() {
        let current = rates(&[("validity_rate", f64::NAN)]);
        let history = vec![rates(&[("validity_rate", 0.98)])];
        let breaches = compute_regression(&current, &history, &thresh(0.9, 0.05));
        assert!(breaches.is_empty(), "expected no breach, got {breaches:?}");
    }

    #[test]
    fn direction_for_maps_lower_is_better_metrics() {
        assert_eq!(
            direction_for("scope_invalid_recall_rate"),
            Direction::LowerIsBetter
        );
        assert_eq!(direction_for("stale_recall_rate"), Direction::LowerIsBetter);
        assert_eq!(direction_for("validity_rate"), Direction::HigherIsBetter);
        assert_eq!(direction_for("unknown_metric"), Direction::HigherIsBetter);
    }

    #[test]
    fn mixed_metrics_report_only_the_breaching_one() {
        let current = rates(&[
            ("validity_rate", 0.85), // floor breach (delta vs 0.88 mean is within tolerance)
            ("scope_invalid_recall_rate", 0.0), // fine
        ]);
        let history = vec![rates(&[
            ("validity_rate", 0.88),
            ("scope_invalid_recall_rate", 0.02),
        ])];
        let breaches = compute_regression(&current, &history, &thresh(0.9, 0.05));
        assert_eq!(breaches.len(), 1);
        assert_eq!(breaches[0].metric, "validity_rate");
        assert_eq!(breaches[0].threshold_type, "floor");
    }
}
