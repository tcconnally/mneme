//! #870: explicit offline/local/provider deployment profiles.
//!
//! One machine-readable, runtime-derived answer to "what is this vault
//! actually connected to?" — model/embedding backends, network listeners,
//! egress, connectors, external mutations, encryption, raw retention.
//!
//! The profile describes ACTUAL runtime state, not configuration intent:
//! the serve handler snapshots the effective flags (offline mode zeroes the
//! web dashboard, LLM, embedding endpoint and connectors at startup), and
//! resolution layers live database state (LLM config, embedding backend,
//! encryption, readiness) on top.
//!
//! Profiles:
//! - `offline`                      — --offline / air-gapped: no listeners
//!   beyond MCP stdio, no egress, no providers, no connectors.
//! - `local_only`                   — all listeners loopback, zero egress.
//! - `local_with_approved_network`  — egress only to operator-configured
//!   endpoints (LLM/embedding providers, connectors) — approved by config.
//! - `external_actions_enabled`     — explicit opt-in to external mutations
//!   (PERSEUS_VAULT_EXTERNAL_ACTIONS=1; off by default — the vault itself
//!   never mutates external systems).
//!
//! Sanitized: hosts only, never URLs, tokens, keys, or raw bodies.

use serde::Serialize;

/// Loopback hosts: localhost, 127.x, [::1], ::1 — port stripped first so
/// `localhost:11434` and `[::1]:8767` classify correctly.
fn is_loopback_host(host: &str) -> bool {
    let h = host_of(host).to_lowercase();
    h == "localhost"
        || h == "::1"
        || h == "[::1]"
        || h.starts_with("127.")
        || h.starts_with("[::1]")
}

/// Extract the host (no port) from an http(s) URL ("" on parse failure).
/// Handles `host`, `host:port`, `[::1]:port`, and userinfo prefixes.
pub(crate) fn host_of(url: &str) -> String {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let hostport = &rest[..end];
    let hostport = hostport
        .rsplit_once('@')
        .map(|(_, h)| h)
        .unwrap_or(hostport);
    // Strip `:port` — but only for non-bracketed hosts (IPv6 `[::1]:8767`
    // keeps its brackets; the port follows the closing bracket).
    if hostport.starts_with('[') {
        if let Some(close) = hostport.find(']') {
            return hostport[..=close].to_string();
        }
        return hostport.to_string();
    }
    // Bare IPv6 (multiple colons, no brackets) has no port to strip.
    if hostport.matches(':').count() == 1 {
        if let Some(i) = hostport.rfind(':') {
            return hostport[..i].to_string();
        }
    }
    hostport.to_string()
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelBackend {
    /// `bundled` | `ollama` | `provider` | `none`
    pub kind: String,
    pub model: String,
    pub available: bool,
    pub degraded: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingBackend {
    /// `bundled` | `provider` | `none`
    pub kind: String,
    pub available: bool,
    pub degraded: bool,
    /// Cross-check from readiness: `available` | `no_coverage` | `disabled`.
    pub semantic_recall: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConnectorInfo {
    pub name: String,
    pub remote: bool,
    pub remote_host: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Network {
    /// Always includes `mcp_stdio`.
    pub listeners: Vec<String>,
    /// Non-loopback egress targets (hosts only, no URLs/keys).
    pub egress_hosts: Vec<String>,
    pub loopback_only: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Encryption {
    /// `aes_256_gcm` | `plaintext`
    pub at_rest: String,
    /// Storage-state probe: `encrypted` | `plaintext` | `mixed-legacy` | `unknown`.
    pub storage_state: String,
    /// `hmac-sha256-blind-token-v1` when an encrypted store has activated the
    /// protected FTS representation; `plaintext` for plaintext stores;
    /// `undeclared` for an encrypted store that still needs migration.
    pub search_index: String,
    /// `loopback_only` | `operator_configured`
    pub in_transit: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawRetention {
    /// The vault persists memory bodies (that IS the store); encrypted at rest.
    pub memory_bodies: String,
    /// Journal/audit records store content digests (sha256), never raw bodies.
    pub raw_logs: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeploymentProfile {
    /// `offline` | `local_only` | `local_with_approved_network` | `external_actions_enabled`
    pub profile: String,
    pub resolved_at_unix_ms: i64,
    pub model_backend: ModelBackend,
    pub embedding_backend: EmbeddingBackend,
    pub network: Network,
    pub connectors: Vec<ConnectorInfo>,
    /// `none` or comma-joined non-loopback egress hosts.
    pub cloud_provider_use: String,
    /// `disabled` | `enabled` (explicit opt-in only).
    pub external_mutations: String,
    pub encryption: Encryption,
    pub raw_retention: RawRetention,
}

/// Runtime context snapshotted by the serve handler at startup (the
/// effective flags AFTER offline-mode zeroing). `external_actions` is also a
/// startup snapshot (PERSEUS_VAULT_EXTERNAL_ACTIONS=1) — deliberately not a
/// live env read, so concurrent callers never race a mutable process global.
#[derive(Debug, Clone, Default)]
pub struct DeploymentContext {
    pub offline: bool,
    pub web_enabled: bool,
    pub web_bind: String,
    pub grpc_enabled: bool,
    pub external_actions: bool,
}

/// One runtime connector's identity, as snapshotted from the Database's
/// loaded connector list.
#[derive(Debug, Clone)]
pub struct ConnectorRuntimeInfo {
    pub name: String,
    pub remote_host: Option<String>,
}

/// Resolve the deployment profile from live database + snapshot context.
/// Pure function of `db` state — no network calls, no side effects.
pub fn resolve(db: &crate::db::Database, ctx: &DeploymentContext) -> DeploymentProfile {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    // ── Model / embedding backends ──────────────────────────────────────
    let llm_enabled = db.llm_enabled();
    let llm_endpoint = db.llm_endpoint();
    let llm_model = db.llm_model();
    let embedding_endpoint = db.llm_embedding_endpoint();

    let (model_kind, model_available, model_degraded) = if !llm_enabled {
        ("none".to_string(), false, false)
    } else if is_loopback_host(&host_of(&llm_endpoint)) {
        ("ollama".to_string(), true, false)
    } else {
        ("provider".to_string(), true, false)
    };

    let emb_kind = db.embedding_kind().to_string();
    let emb_configured = emb_kind != "none";
    // Degraded = configured but not actually usable (provider endpoint set
    // while the LLM integration is off; bundled backend disabled). Reported
    // explicitly — never silently reclassified as empty success.
    let emb_degraded = emb_configured
        && ((emb_kind == "provider" && !llm_enabled)
            || (emb_kind == "bundled" && !db.embedding_enabled()));
    let emb_available = match emb_kind.as_str() {
        "bundled" => db.embedding_enabled(),
        // A provider endpoint is only usable when the LLM integration that
        // would call it is actually enabled.
        "provider" => llm_enabled && !embedding_endpoint.as_deref().unwrap_or("").is_empty(),
        _ => false,
    };
    let semantic_recall = db.readiness().semantic_recall().to_string();

    // ── Network listeners ───────────────────────────────────────────────
    let mut listeners = vec!["mcp_stdio".to_string()];
    if !ctx.offline && ctx.web_enabled {
        listeners.push(format!("web_dashboard({})", ctx.web_bind));
    }
    if !ctx.offline && ctx.grpc_enabled {
        listeners.push("grpc".to_string());
    }

    // ── Egress ──────────────────────────────────────────────────────────
    let mut egress: Vec<String> = Vec::new();
    let mut push_egress = |host: Option<String>| {
        if let Some(h) = host {
            if !h.is_empty() && !is_loopback_host(&h) && !egress.contains(&h) {
                egress.push(h);
            }
        }
    };
    if llm_enabled {
        push_egress(Some(host_of(&llm_endpoint)));
    }
    if let Some(ep) = embedding_endpoint {
        push_egress(Some(host_of(&ep)));
    }
    let mut connectors: Vec<ConnectorInfo> = Vec::new();
    for c in db.connectors_snapshot() {
        let remote = c.remote_host.is_some();
        let host = c.remote_host.unwrap_or_default();
        push_egress(if remote { Some(host.clone()) } else { None });
        connectors.push(ConnectorInfo {
            name: c.name,
            remote,
            remote_host: host,
        });
    }
    // ── Profile classification ──────────────────────────────────────────
    let external_actions = ctx.external_actions;
    let loopback_only = egress.is_empty()
        && listeners.iter().all(|l| {
            // mcp_stdio is always local; web/grpc listeners carry their bind.
            !l.contains("0.0.0.0") && !l.contains("[::]")
        });
    let profile = if ctx.offline {
        "offline"
    } else if external_actions {
        "external_actions_enabled"
    } else if egress.is_empty() {
        "local_only"
    } else {
        "local_with_approved_network"
    };

    // ── Encryption / retention ──────────────────────────────────────────
    let storage_state = db.encryption_storage_state();
    let at_rest = if db.encryption_enabled()
        || matches!(
            storage_state.as_str(),
            "encrypted" | "encrypted-incomplete" | "mixed-legacy"
        ) {
        "aes_256_gcm"
    } else {
        "plaintext"
    };
    let search_index = if let Some(mode) = db.encryption_search_mode() {
        mode
    } else if at_rest == "aes_256_gcm" {
        "undeclared".to_string()
    } else {
        "plaintext".to_string()
    };
    let in_transit = if loopback_only {
        "loopback_only"
    } else {
        "operator_configured"
    };

    // Cloud-provider summary (computed before the struct literal consumes
    // `connectors`): non-loopback egress hosts from providers + connectors.
    let cloud_hosts: Vec<String> = {
        let mut hosts: Vec<String> = db
            .llm_embedding_endpoint()
            .map(|e| host_of(&e))
            .filter(|h| !is_loopback_host(h))
            .into_iter()
            .collect();
        if llm_enabled {
            let h = host_of(&llm_endpoint);
            if !is_loopback_host(&h) && !hosts.contains(&h) {
                hosts.push(h);
            }
        }
        for c in &connectors {
            if c.remote && !c.remote_host.is_empty() && !hosts.contains(&c.remote_host) {
                hosts.push(c.remote_host.clone());
            }
        }
        hosts
    };

    DeploymentProfile {
        profile: profile.to_string(),
        resolved_at_unix_ms: now,
        model_backend: ModelBackend {
            kind: model_kind,
            model: llm_model,
            available: model_available,
            degraded: model_degraded,
        },
        embedding_backend: EmbeddingBackend {
            kind: emb_kind,
            available: emb_available,
            degraded: emb_degraded,
            semantic_recall,
        },
        network: Network {
            listeners,
            egress_hosts: egress,
            loopback_only,
        },
        connectors,
        cloud_provider_use: if cloud_hosts.is_empty() {
            "none".to_string()
        } else {
            cloud_hosts.join(",")
        },
        external_mutations: if external_actions {
            "enabled".to_string()
        } else {
            "disabled".to_string()
        },
        encryption: Encryption {
            at_rest: at_rest.to_string(),
            storage_state,
            search_index,
            in_transit: in_transit.to_string(),
        },
        raw_retention: RawRetention {
            memory_bodies: "retained_at_rest".to_string(),
            raw_logs: "digest_only".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_extraction_and_loopback() {
        assert_eq!(host_of("http://localhost:11434/api/generate"), "localhost");
        assert_eq!(
            host_of("https://api.openai.com/v1/embeddings"),
            "api.openai.com"
        );
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("127.8.8.8"));
        assert!(is_loopback_host("::1"));
        assert!(!is_loopback_host("10.168.168.66"));
        assert!(!is_loopback_host("api.github.com"));
    }
}

// ─── #870 integration tests (tools-level, via the Database) ───────────────

#[cfg(test)]
mod integration {
    use super::*;
    use crate::tools::{handle_deployment_profile, handle_health};
    use serde_json::Value;

    fn temp_db() -> (crate::db::TestDatabase, String) {
        let db = crate::db::TestDatabase::new("deployment_profile");
        let path = db.path().to_string();
        (db, path)
    }

    #[test]
    fn offline_profile_zeroes_egress_and_recall_paths_run_local() {
        let (mut db, path) = temp_db();
        db.set_deployment_context(true, true, "0.0.0.0", false, false);
        let p = resolve(&db, db.deployment_context());
        assert_eq!(p.profile, "offline");
        assert_eq!(
            p.network.listeners,
            vec!["mcp_stdio"],
            "offline: web disabled"
        );
        assert!(p.network.egress_hosts.is_empty(), "offline: no egress");
        assert!(p.connectors.is_empty());
        assert_eq!(p.external_mutations, "disabled");
        assert!(p.network.loopback_only);
        // Offline no-network acceptance: every recall/maintenance path the
        // profile covers must complete with ZERO network calls (they are all
        // local SQLite/bundled paths — in this sandbox there is no network,
        // so success IS the proof no egress is attempted).
        let r = crate::models::RecallParams {
            query: "offline probe".to_string(),
            limit: 5,
            mode: crate::models::SearchMode::Fts5,
            ..Default::default()
        };
        assert!(db.recall(&r).is_ok(), "fts5 in offline");
        #[cfg(feature = "bundled-embeddings")]
        {
            let mut h = r.clone();
            h.mode = crate::models::SearchMode::Hybrid;
            assert!(db.recall(&h).is_ok(), "hybrid (bundled) in offline");
        }
        // Fused covers the graph + temporal arms (default strategies; the
        // dense arm degrades gracefully on lite builds).
        let mut f = r.clone();
        f.mode = crate::models::SearchMode::Fused;
        assert!(
            db.recall(&f).is_ok(),
            "fused (graph/temporal/dense) in offline"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn local_only_when_no_egress_and_loopback_listeners() {
        let (mut db, path) = temp_db();
        db.set_deployment_context(false, true, "127.0.0.1", false, false);
        let p = resolve(&db, db.deployment_context());
        assert_eq!(p.profile, "local_only");
        assert!(p
            .network
            .listeners
            .contains(&"web_dashboard(127.0.0.1)".to_string()));
        assert!(p.network.loopback_only);
        assert_eq!(p.cloud_provider_use, "none");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn approved_network_profile_reports_egress_hosts_sanitized() {
        let (mut db, path) = temp_db();
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        db.set_llm(
            true,
            "http://10.0.0.5:11434/api/generate",
            "qwen3.5:9b",
            None,
            Some("http://10.0.0.5:11434/api/embed"),
            None,
        );
        let p = resolve(&db, db.deployment_context());
        assert_eq!(p.profile, "local_with_approved_network");
        assert!(p.network.egress_hosts.contains(&"10.0.0.5".to_string()));
        assert!(p.cloud_provider_use.contains("10.0.0.5"));
        assert!(
            !p.cloud_provider_use.contains("api/generate"),
            "sanitized: no URL paths"
        );
        assert_eq!(p.model_backend.kind, "provider");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn external_actions_requires_explicit_opt_in() {
        let (mut db, path) = temp_db();
        db.set_deployment_context(false, false, "127.0.0.1", false, true);
        let p = resolve(&db, db.deployment_context());
        assert_eq!(p.external_mutations, "enabled");
        assert_eq!(p.profile, "external_actions_enabled");
        let (mut db2, path2) = temp_db();
        db2.set_deployment_context(false, false, "127.0.0.1", false, false);
        let p2 = resolve(&db2, db2.deployment_context());
        assert_eq!(p2.external_mutations, "disabled");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&path2);
    }

    #[test]
    fn local_provider_misconfig_is_degraded_not_empty_success() {
        // Acceptance: a missing/unavailable local backend is reported as
        // degraded, never silently reclassified as empty success.
        let (mut db, path) = temp_db();
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        // Provider endpoint configured while the LLM integration is off —
        // the embedding backend cannot actually be reached.
        db.set_llm(
            false,
            "http://127.0.0.1:9/api/generate",
            "unused",
            None,
            Some("http://127.0.0.1:9/v1/embeddings"),
            None,
        );
        #[cfg(not(feature = "bundled-embeddings"))]
        {
            // Lite build: the provider IS the only embedding backend.
            assert_eq!(db.embedding_kind(), "provider");
            let p = resolve(&db, db.deployment_context());
            assert!(
                p.embedding_backend.degraded,
                "provider configured but LLM off must be degraded: {p:?}"
            );
            assert!(!p.embedding_backend.available);
        }
        #[cfg(feature = "bundled-embeddings")]
        {
            // Bundled build: the local backend is the source of truth and
            // remains available (not silently reclassified).
            assert_eq!(db.embedding_kind(), "bundled");
            let p = resolve(&db, db.deployment_context());
            assert!(p.embedding_backend.available, "{p:?}");
            // Backend on; the fresh test store has nothing embedded yet.
            assert_eq!(p.embedding_backend.semantic_recall, "no_coverage", "{p:?}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn profile_reports_encryption_and_retention_fields() {
        let (mut db, path) = temp_db();
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        let p = resolve(&db, db.deployment_context());
        assert!(matches!(
            p.encryption.at_rest.as_str(),
            "aes_256_gcm" | "plaintext"
        ));
        assert!(!p.encryption.storage_state.is_empty());
        assert!(p.raw_retention.memory_bodies.contains("retained_at_rest"));
        assert_eq!(p.raw_retention.raw_logs, "digest_only");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn incomplete_encryption_is_not_reported_as_plaintext_after_reopen() {
        let (mut db, path) = temp_db();
        let key_path = std::env::temp_dir().join(format!(
            "perseus-vault-profile-key-{}.key",
            uuid::Uuid::new_v4()
        ));
        let key = crate::encryption::EncryptionManager::generate_key();
        std::fs::write(&key_path, key).unwrap();
        let key_path = key_path.to_string_lossy().into_owned();
        db.set_encryption(&key_path).unwrap();
        db.remember(&crate::db::tests::make_entity(
            "profile-incomplete",
            "facts",
            "profile-incomplete",
            r#"{"note":"profile state"}"#,
        ))
        .unwrap();
        db.conn()
            .unwrap()
            .execute("DELETE FROM entities_fts", [])
            .unwrap();

        let reopened = crate::db::Database::open(&path).unwrap();
        let profile = resolve(&reopened, reopened.deployment_context());
        assert_eq!(profile.encryption.storage_state, "encrypted-incomplete");
        assert_eq!(profile.encryption.at_rest, "aes_256_gcm");
        assert_eq!(profile.encryption.search_index, "undeclared");

        let _ = std::fs::remove_file(&key_path);
        drop(reopened);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tool_and_health_expose_the_resolved_profile() {
        let (mut db, path) = temp_db();
        db.set_deployment_context(false, false, "127.0.0.1", false, false);
        let raw = handle_deployment_profile(&db, serde_json::json!({})).expect("profile tool");
        let v: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(v["profile"], "local_only");
        assert!(v["network"]["listeners"]
            .as_array()
            .unwrap()
            .contains(&"mcp_stdio".into()));
        let health = handle_health(&db);
        let hv: Value = serde_json::from_str(&health).unwrap();
        assert_eq!(hv["deployment_profile"]["profile"], "local_only");
        assert_eq!(hv["status"], "healthy");
        let _ = std::fs::remove_file(&path);
    }
}
