//! #1010: per-stage provider/config self-report with a resolved-vs-requested
//! diff (borrowed from Hy-Memory + MindCache practitioner scans — see
//! insight/hy-memory-competitive-scan and insight/public-repo-comparison-
//! mindcache-perseus).
//!
//! The lesson from both systems: an operator requests one configuration and
//! the runtime silently resolves another (mode-vs-config drift; a stage that
//! hardcodes a default provider while a non-default is configured, logging
//! "LLM returned empty" and continuing). This module makes drift a loud,
//! machine-readable condition: every stage reports WHAT WAS REQUESTED (the
//! operator-facing knob as literally given), WHAT ACTUALLY RESOLVED (the
//! runtime state), and a `drifted` flag with a remediation note.
//!
//! Extends #870's deployment profile (resolved posture only) with the
//! requested half of the diff.

use serde::Serialize;

use crate::db::Database;
use crate::deployment_profile::{self, DeploymentContext};

#[derive(Debug, Clone, Serialize)]
pub struct StageReport {
    /// Stable stage id: `embedding_backend` | `model_backend` |
    /// `quantization` | `db_path` | `encryption` | `network`.
    pub stage: String,
    /// The operator-facing knob as literally given. Sanitized: no secrets,
    /// keys, or full URLs — hosts and kind labels only.
    pub requested: String,
    /// What the runtime actually uses (the #870 resolved posture).
    pub resolved: String,
    /// True when resolved != requested in a way the operator did not ask for.
    pub drifted: bool,
    /// Drift explanation + remediation ("" when not drifted).
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigReport {
    pub generated_at_unix_ms: i64,
    pub stages: Vec<StageReport>,
    /// Stage ids with `drifted == true`. Empty = every stage resolved exactly
    /// as requested.
    pub drifted_stages: Vec<String>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Pure function of `db` state + snapshot context + the live environment
/// (env knobs are process-static for the vault's lifetime). No network calls.
pub fn build(db: &Database, ctx: &DeploymentContext) -> ConfigReport {
    let profile = deployment_profile::resolve(db, ctx);
    let mut stages: Vec<StageReport> = Vec::new();

    // ── embedding backend ────────────────────────────────────────────────
    {
        let kind = db.embedding_kind();
        let endpoint = db.llm_embedding_endpoint();
        let requested = match kind.as_ref() {
            "provider" => endpoint
                .map(|e| format!("provider endpoint ({})", deployment_profile::host_of(&e)))
                .unwrap_or_else(|| "provider endpoint (unset)".to_string()),
            "bundled" => "bundled local model".to_string(),
            _ => "none".to_string(),
        };
        let resolved = format!(
            "kind={} available={} degraded={} semantic_recall={}",
            profile.embedding_backend.kind,
            profile.embedding_backend.available,
            profile.embedding_backend.degraded,
            profile.embedding_backend.semantic_recall,
        );
        let drifted = profile.embedding_backend.degraded;
        stages.push(StageReport {
            stage: "embedding_backend".to_string(),
            requested,
            resolved,
            drifted,
            note: if drifted {
                "configured embedding backend is not usable (provider endpoint set while \
                 the LLM integration is off, or bundled backend disabled); dense recall is \
                 degraded — never treated as empty success"
                    .to_string()
            } else {
                String::new()
            },
        });
    }

    // ── model backend (LLM synthesizer) ─────────────────────────────────
    {
        let requested = if db.llm_enabled() {
            format!(
                "{} ({})",
                deployment_profile::host_of(&db.llm_endpoint()),
                db.llm_model()
            )
        } else {
            "none (llm integration disabled)".to_string()
        };
        let resolved = format!(
            "kind={} available={}",
            profile.model_backend.kind, profile.model_backend.available
        );
        let drifted = profile.model_backend.kind != "none" && !profile.model_backend.available;
        stages.push(StageReport {
            stage: "model_backend".to_string(),
            requested,
            resolved,
            drifted,
            note: if drifted {
                "an LLM endpoint is configured but unavailable — synthesis/dream paths will \
                 fail at call time rather than silently return empty"
                    .to_string()
            } else {
                String::new()
            },
        });
    }

    // ── quantization ─────────────────────────────────────────────────────
    {
        let flag = std::env::var("PERSEUS_VAULT_EMBEDDING_QUANT").ok();
        // Live store-record read (not the open-time cache): the report is a
        // point-in-time diagnostic, and the record is the source of truth
        // for what dense recall decodes.
        let record_q = {
            let conn = db.conn().ok();
            conn.and_then(|c| {
                c.query_row(
                    "SELECT format FROM embedding_format WHERE id = 1",
                    [],
                    |r| r.get::<_, String>(0),
                )
                .ok()
            })
            .unwrap_or_else(|| "float32".to_string())
        };
        let requested = flag.clone().unwrap_or_else(|| {
            "(unset → store embedding_format record, default float32)".to_string()
        });
        let resolved = record_q.clone();
        // Drift when the store resolves to a format the operator did not ask
        // for in THIS process: flag unset while the store record is quantized
        // (a previous process declared it) — or flag set while the store
        // resolved differently (can only happen via direct store edits;
        // normal opens fail hard on mismatch, which is the loud path).
        let drifted = match flag {
            None => resolved != "float32",
            Some(f) => f != resolved,
        };
        stages.push(StageReport {
            stage: "quantization".to_string(),
            requested,
            resolved,
            drifted,
            note: if drifted {
                "the store's embedding format does not match this process's request; dense \
                 recall uses the resolved format — reindex via perseus_vault_embed quant_mode \
                 to change it"
                    .to_string()
            } else {
                String::new()
            },
        });
    }

    // ── fingerprint tier (#1020) ────────────────────────────────────────
    {
        let flag = std::env::var("PERSEUS_VAULT_EMBEDDING_FINGERPRINT").ok();
        let enabled = db.fingerprint_enabled();
        let requested = flag.clone().unwrap_or_else(|| "(unset → off)".to_string());
        let resolved = if enabled {
            "on".to_string()
        } else {
            "off".to_string()
        };
        let drifted = flag
            .as_deref()
            .is_some_and(|f| crate::db::Database::parse_fingerprint_flag(f).ok() != Some(enabled));
        stages.push(StageReport {
            stage: "fingerprint_tier".to_string(),
            requested,
            resolved,
            drifted,
            note: if enabled {
                "dense recall falls back to deterministic fingerprint (Hamming) ranking when \
                 the embedding backend is unavailable"
                    .to_string()
            } else {
                String::new()
            },
        });
    }

    // ── db path ──────────────────────────────────────────────────────────
    {
        let requested_env = std::env::var("PERSEUS_VAULT_DB_PATH").ok();
        let resolved_path = db.db_path();
        let requested = requested_env
            .clone()
            .unwrap_or_else(|| "(default home dir)".to_string());
        let drifted = requested_env
            .as_deref()
            .map(|r| r != resolved_path)
            .unwrap_or(false);
        stages.push(StageReport {
            stage: "db_path".to_string(),
            requested,
            resolved: resolved_path.to_string(),
            drifted,
            note: if drifted {
                "PERSEUS_VAULT_DB_PATH could not be used; the runtime fell back to the \
                 default path — inspect permissions and reopen"
                    .to_string()
            } else {
                String::new()
            },
        });
    }

    // ── encryption at rest ───────────────────────────────────────────────
    {
        let plaintext_allowed = std::env::var("PERSEUS_VAULT_ALLOW_PLAINTEXT")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        let requested = if plaintext_allowed {
            "plaintext allowed (PERSEUS_VAULT_ALLOW_PLAINTEXT set)".to_string()
        } else {
            "encrypted (default)".to_string()
        };
        let resolved = format!(
            "at_rest={} storage_state={}",
            profile.encryption.at_rest, profile.encryption.storage_state
        );
        let drifted = profile.encryption.at_rest == "plaintext"
            || profile.encryption.storage_state == "mixed-legacy";
        stages.push(StageReport {
            stage: "encryption".to_string(),
            requested,
            resolved,
            drifted,
            note: if drifted {
                "the store is running with encryption off or a mixed legacy/encrypted \
                 state — loud by design; run rekey/migration tooling to return to \
                 aes_256_gcm"
                    .to_string()
            } else {
                String::new()
            },
        });
    }

    // ── network listeners ────────────────────────────────────────────────
    {
        // Effective-snapshot stage: ctx holds the flags AFTER startup
        // zeroing, so requested == resolved by construction; drift cannot
        // occur here and the stage exists for one-query completeness.
        let requested = format!(
            "web_enabled={} grpc_enabled={} offline={}",
            ctx.web_enabled, ctx.grpc_enabled, ctx.offline
        );
        let resolved = profile.network.listeners.join(",");
        stages.push(StageReport {
            stage: "network".to_string(),
            requested,
            resolved,
            drifted: false,
            note: "effective snapshot after startup zeroing (offline mode zeroes listeners \
                   before this report is built)"
                .to_string(),
        });
    }

    let drifted_stages: Vec<String> = stages
        .iter()
        .filter(|s| s.drifted)
        .map(|s| s.stage.clone())
        .collect();
    ConfigReport {
        generated_at_unix_ms: now_ms(),
        stages,
        drifted_stages,
    }
}

/// One-line-per-stage log block for startup; drift lines are loud.
pub fn log_block(db: &Database, ctx: &DeploymentContext) {
    let report = build(db, ctx);
    for s in &report.stages {
        if s.drifted {
            eprintln!(
                "perseus-vault: CONFIG DRIFT [{stage}]: requested=[{requested}] resolved=[{resolved}] note=[{note}]",
                stage = s.stage,
                requested = s.requested,
                resolved = s.resolved,
                note = s.note,
            );
        } else {
            eprintln!(
                "perseus-vault: config [{stage}]: requested=[{requested}] resolved=[{resolved}]",
                stage = s.stage,
                requested = s.requested,
                resolved = s.resolved,
            );
        }
    }
    if !report.drifted_stages.is_empty() {
        eprintln!(
            "perseus-vault: CONFIG DRIFT in stages: {} — query perseus_vault_config_report for \
             the machine-readable diff",
            report.drifted_stages.join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::TestDatabase;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn stage<'a>(r: &'a ConfigReport, name: &str) -> &'a StageReport {
        r.stages
            .iter()
            .find(|s| s.stage == name)
            .unwrap_or_else(|| panic!("missing stage {name}: {:?}", r.stages))
    }

    #[test]
    fn default_config_reports_all_stages_without_drift() {
        let mut db = TestDatabase::new("cfg-report");
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        let r = build(&db, db.deployment_context());
        assert_eq!(r.stages.len(), 7);
        // Fresh store: quantization resolves to float32 (the store default).
        assert_eq!(stage(&r, "quantization").resolved, "float32");
        assert!(!stage(&r, "quantization").drifted);
        // Fingerprint tier defaults off with no drift.
        assert_eq!(stage(&r, "fingerprint_tier").resolved, "off");
        assert!(!stage(&r, "fingerprint_tier").drifted);
        // The test harness opens stores WITHOUT encryption (plaintext), which
        // the report correctly flags as drift — loud by design. The other
        // stages must resolve cleanly.
        let unexpected: Vec<&str> = r
            .drifted_stages
            .iter()
            .map(|s| s.as_str())
            .filter(|s| *s != "encryption")
            .collect();
        assert!(
            unexpected.is_empty(),
            "unexpected drift in a default config: {unexpected:?}"
        );
    }

    #[test]
    fn degraded_embedding_backend_drifts_loudly() {
        let mut db = TestDatabase::new("cfg-report-degraded");
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        // Provider endpoint configured while the LLM integration is off:
        // the embedding backend cannot be reached — the exact Hy-Memory /
        // MindCache silent-drift scenario this report exists for.
        db.set_llm(
            false,
            "http://127.0.0.1:9/api/generate",
            "unused",
            None,
            Some("http://127.0.0.1:9/v1/embeddings"),
            None,
        );
        let r = build(&db, db.deployment_context());
        #[cfg(feature = "bundled-embeddings")]
        {
            // Bundled build: local backend remains the source of truth —
            // no drift (the misconfigured provider does not override it).
            assert!(!stage(&r, "embedding_backend").drifted);
        }
        #[cfg(not(feature = "bundled-embeddings"))]
        {
            assert!(stage(&r, "embedding_backend").drifted);
            assert!(r.drifted_stages.contains(&"embedding_backend".to_string()));
            assert!(!stage(&r, "embedding_backend").note.is_empty());
        }
    }

    #[test]
    fn quantized_store_without_flag_drifts_against_default() {
        let mut db = TestDatabase::new("cfg-report-quant");
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        // Simulate a store whose embedding_format record was declared by a
        // previous process: this process requests nothing (default float32)
        // but the store resolves int8 — a real requested-vs-resolved diff.
        let conn = db.conn().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO embedding_format (id, format, updated_at_unix_ms) \
             VALUES (1, 'int8', 0)",
            [],
        )
        .unwrap();
        drop(conn);
        let r = build(&db, db.deployment_context());
        let q = stage(&r, "quantization");
        assert_eq!(q.resolved, "int8");
        assert!(q.drifted, "{q:?}");
        assert!(q.note.contains("reindex"));
    }

    // #1020: the fingerprint tier reports its resolved state (off by
    // default, no drift, no note).
    #[test]
    fn fingerprint_tier_stage_reports_default_off_without_drift() {
        // The sibling fingerprint tests mutate PERSEUS_VAULT_EMBEDDING_FINGERPRINT
        // under ENV_LOCK; take it here too so a concurrent set_var can't make
        // the default-off resolution look drifted (macOS scheduler race).
        let _guard = ENV_LOCK.lock().unwrap();
        let mut db = TestDatabase::new("cfg-report-fp");
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        let r = build(&db, db.deployment_context());
        let f = stage(&r, "fingerprint_tier");
        assert_eq!(f.resolved, "off");
        assert!(!f.drifted, "{f:?}");
        assert!(f.note.is_empty());
    }

    // #1020: a process-level flag that disagrees with the open-time
    // resolution is a loud requested-vs-resolved diff.
    #[test]
    fn fingerprint_flag_mismatch_drifts() {
        let _guard = ENV_LOCK.lock().unwrap();
        let mut db = TestDatabase::new("cfg-report-fp-drift");
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        // The Database resolved OFF at open (flag unset); now request ON in
        // this process — exactly the drift condition this stage reports.
        std::env::set_var("PERSEUS_VAULT_EMBEDDING_FINGERPRINT", "on");
        let r = build(&db, db.deployment_context());
        std::env::remove_var("PERSEUS_VAULT_EMBEDDING_FINGERPRINT");
        let f = stage(&r, "fingerprint_tier");
        assert_eq!(f.requested, "on");
        assert_eq!(f.resolved, "off");
        assert!(f.drifted, "{f:?}");
    }

    #[test]
    fn db_path_env_mismatch_drifts() {
        let _guard = ENV_LOCK.lock().unwrap();
        let mut db = TestDatabase::new("cfg-report-path");
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        // Point the knob at a path the runtime is not actually using.
        std::env::set_var("PERSEUS_VAULT_DB_PATH", "/nonexistent/forced.db");
        let r = build(&db, db.deployment_context());
        let p = stage(&r, "db_path");
        assert!(p.drifted, "{p:?}");
        assert!(p.note.contains("PERSEUS_VAULT_DB_PATH"));
        std::env::remove_var("PERSEUS_VAULT_DB_PATH");
    }

    #[test]
    fn report_is_machine_readable_and_stage_stable() {
        let mut db = TestDatabase::new("cfg-report-shape");
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        let r = build(&db, db.deployment_context());
        let json = serde_json::to_string(&r).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["generated_at_unix_ms"].is_number());
        assert!(v["stages"].is_array());
        assert!(v["drifted_stages"].is_array());
        for s in v["stages"].as_array().unwrap() {
            assert!(s["stage"].is_string());
            assert!(s["requested"].is_string());
            assert!(s["resolved"].is_string());
            assert!(s["drifted"].is_boolean());
            assert!(s["note"].is_string());
        }
    }
}
