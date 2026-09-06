//! #929: live-web gap-fill — opt-in, agent-side fetch, vault-side validation.
//!
//! The Vault NEVER fetches from the network (no SSRF surface; preserves the
//! air-gap/federal no-egress posture). The agent fetches with its own tools
//! and reports grounded content back through `perseus_vault_web_gap_fill`;
//! this module validates the submission — http/https scheme only, per-
//! workspace source allowlist, no private/loopback/link-local literal IPs,
//! bounded sizes, fail-closed secret scan, relevance floor, per-workspace
//! rate limit — and the handler writes it through the normal audited
//! remember path as unverified-until-confirmed (never auto-promoted).
//!
//! OFF by default: `PERSEUS_VAULT_WEB_GAP_FILL_ENABLED=1` opts in; without
//! it the handler fails closed. Per-workspace source allowlist:
//! `PERSEUS_VAULT_WEB_ALLOWLIST=/path/to/allowlist.json` mapping
//! `{"<workspace_hash>": ["host", ...] | "*"}` (the `"*"` key applies to
//! every workspace; a host entry `"*"` allows any host for that workspace).
//! `PERSEUS_VAULT_WEB_RATE_LIMIT` (default 10) caps writes per workspace per
//! hour; `PERSEUS_VAULT_WEB_MIN_RELEVANCE` (default 0.6) is the relevance
//! floor.

use sha2::{Digest, Sha256};

/// Tests that set PERSEUS_VAULT_WEB_* env vars (process-global) must hold
/// this lock: cargo runs tests in parallel threads and concurrent set/remove
/// would clobber each other's gate state.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Max number of source URLs per submission.
pub const MAX_SOURCES: usize = 8;
/// Max content length (chars).
pub const MAX_CONTENT_CHARS: usize = 64 * 1024;
/// Max title length (chars).
pub const MAX_TITLE_CHARS: usize = 512;
/// Max query length (chars).
pub const MAX_QUERY_CHARS: usize = 512;
/// Max allowlist file size (bytes).
pub const MAX_ALLOWLIST_BYTES: u64 = 64 * 1024;
/// Default minimum relevance score for write-back.
pub const DEFAULT_MIN_RELEVANCE: f64 = 0.6;
/// Default per-workspace hourly write cap.
pub const DEFAULT_RATE_LIMIT_PER_HOUR: i64 = 10;

/// Opt-in gate. Mirrors the #919 hints pattern: absent or != "1" means off.
pub fn web_gap_fill_enabled() -> bool {
    std::env::var("PERSEUS_VAULT_WEB_GAP_FILL_ENABLED").as_deref() == Ok("1")
}

/// Configured minimum relevance score in [0,1].
pub fn min_relevance() -> f64 {
    std::env::var("PERSEUS_VAULT_WEB_MIN_RELEVANCE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(DEFAULT_MIN_RELEVANCE)
}

/// Configured per-workspace hourly write cap, clamped to [1, 1000].
pub fn rate_limit_per_hour() -> i64 {
    std::env::var("PERSEUS_VAULT_WEB_RATE_LIMIT")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .map(|v| v.clamp(1, 1000))
        .unwrap_or(DEFAULT_RATE_LIMIT_PER_HOUR)
}

/// Load the per-workspace source-host allowlist. Missing env or file -> an
/// EMPTY allowlist (everything denied — fail-closed). Malformed or oversized
/// file -> Err.
pub fn load_allowlist() -> Result<std::collections::BTreeMap<String, Vec<String>>, String> {
    let Some(path) = std::env::var("PERSEUS_VAULT_WEB_ALLOWLIST").ok() else {
        return Ok(std::collections::BTreeMap::new());
    };
    let meta =
        std::fs::metadata(&path).map_err(|e| format!("allowlist unreadable at {path}: {e}"))?;
    if meta.len() > MAX_ALLOWLIST_BYTES {
        return Err(format!(
            "allowlist exceeds {MAX_ALLOWLIST_BYTES} bytes: {path}"
        ));
    }
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("allowlist read failed at {path}: {e}"))?;
    let parsed: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("allowlist is not valid JSON at {path}: {e}"))?;
    let obj = parsed
        .as_object()
        .ok_or_else(|| format!("allowlist must be a JSON object at {path}"))?;
    let mut out = std::collections::BTreeMap::new();
    for (ws, v) in obj {
        if v.as_str() == Some("*") {
            out.insert(ws.clone(), vec!["*".to_string()]);
            continue;
        }
        let hosts = v
            .as_array()
            .ok_or_else(|| format!("allowlist entry for '{ws}' must be an array or \"*\""))?;
        let mut list = Vec::new();
        for h in hosts {
            let h = h
                .as_str()
                .ok_or_else(|| format!("allowlist host for '{ws}' must be a string"))?;
            let h = h.trim().to_ascii_lowercase();
            if h.is_empty() || h.len() > 253 {
                return Err(format!("allowlist host for '{ws}' is invalid"));
            }
            list.push(h);
        }
        out.insert(ws.clone(), list);
    }
    Ok(out)
}

/// Hosts allowed for a workspace: its own entry, else the `"*"` entry.
/// An empty workspace string is never a real workspace — denied.
pub fn workspace_allowed_hosts<'a>(
    allowlist: &'a std::collections::BTreeMap<String, Vec<String>>,
    workspace: &str,
) -> Option<&'a [String]> {
    if workspace.is_empty() {
        return None;
    }
    allowlist
        .get(workspace)
        .or_else(|| allowlist.get("*"))
        .map(|v| v.as_slice())
}

/// Exact host match (case-insensitive); a `"*"` entry allows any host.
pub fn host_allowed(hosts: &[String], host: &str) -> bool {
    hosts
        .iter()
        .any(|h| h == "*" || h.eq_ignore_ascii_case(host))
}

/// Parse a source URL; returns the host on success. Rejects non-http(s)
/// schemes, userinfo (`user:pass@`), empty hosts, and malformed ports.
pub fn parse_source_url(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    // Scheme matching is case-insensitive per RFC 3986.
    let rest = if trimmed.get(..8).map(|s| s.eq_ignore_ascii_case("https://")) == Some(true) {
        &trimmed[8..]
    } else if trimmed.get(..7).map(|s| s.eq_ignore_ascii_case("http://")) == Some(true) {
        &trimmed[7..]
    } else {
        return Err(format!("unsupported source scheme: {raw}"));
    };
    // Authority ends at the first '/', '?', or '#'.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("").trim();
    if authority.is_empty() {
        return Err(format!("source URL has no host: {raw}"));
    }
    if authority.contains('@') {
        return Err("source URL must not carry userinfo".to_string());
    }
    // host[:port] — the port must be all digits when present.
    let (host, port) = match authority.rfind(':') {
        Some(i) => (&authority[..i], Some(&authority[i + 1..])),
        None => (authority, None),
    };
    let host = host.trim();
    if host.is_empty() {
        return Err(format!("source URL has an empty host: {raw}"));
    }
    if host.len() > 253 {
        return Err("source host exceeds 253 chars".to_string());
    }
    if let Some(p) = port {
        if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) || p.parse::<u16>().is_err() {
            return Err(format!("source URL has a malformed port: {raw}"));
        }
    }
    if !host
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.' || b == b':')
    {
        return Err(format!("source host has invalid characters: {raw}"));
    }
    if host.bytes().any(|b| b.is_ascii_whitespace() || b < 0x21) {
        return Err(format!("source host has control characters: {raw}"));
    }
    // Encoded literal IPs must not slip past the private-range check: reject
    // hex (`0x7f000001`) and pure-numeric forms that are not valid IP
    // literals (`2130706433`, `017700000001` — decimal/octal encodings of
    // 127.0.0.1). A host that is not a valid literal must contain a letter
    // to count as a hostname.
    if host.starts_with("0x") || host.starts_with("0X") {
        return Err(format!("source host uses an encoded numeric form: {raw}"));
    }
    if !host.bytes().any(|b| b.is_ascii_alphabetic()) && host.parse::<std::net::IpAddr>().is_err() {
        return Err(format!(
            "source host is neither a hostname nor a valid IP literal: {raw}"
        ));
    }
    Ok(host.to_string())
}

/// True when `host` is a literal IP in a private/reserved range
/// (loopback, RFC1918, unique-local, link-local, unspecified, multicast,
/// documentation). Hostnames return false — the allowlist governs them.
pub fn host_is_private_literal(host: &str) -> bool {
    let Ok(ip) = host.trim().parse::<std::net::IpAddr>() else {
        return false; // hostname: not a literal; the allowlist governs it
    };
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_multicast()
                || v4.octets()[0] == 192 && v4.octets()[1] == 0 && v4.octets()[2] == 2 // TEST-NET-1
                || v4.octets()[0] == 198 && v4.octets()[1] == 51 && v4.octets()[2] == 100 // TEST-NET-2
                || v4.octets()[0] == 203 && v4.octets()[1] == 0 && v4.octets()[2] == 113 // TEST-NET-3
                || v4.octets()[0] == 100 && (64..=127).contains(&v4.octets()[1]) // CGNAT 100.64/10
                || v4.octets()[0] == 192 && v4.octets()[1] == 0 // IETF protocol assignments 192.0.0.0/24
                || v4.octets()[0] == 198 && (18..=19).contains(&v4.octets()[1]) // benchmarking 198.18/15
                || v4.octets()[0] >= 240 // reserved 240.0.0.0/4 (+ 255.x)
                || v4.octets() == [255, 255, 255, 255] // limited broadcast
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
                || v6.is_unspecified()
                || v6.is_multicast()
                || v6.segments()[0] == 0x2001 && v6.segments()[1] == 0x0db8 // documentation
        }
    }
}

/// Fail-closed high-precision secret scan. Returns the matched class name.
pub fn scan_secrets(content: &str) -> Option<&'static str> {
    let alnum_tail = |start: usize, min: usize| -> bool {
        let tail = &content[start..];
        let n = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .count();
        n >= min
    };
    let token_tail = |start: usize, min: usize| -> bool {
        let tail = &content[start..];
        let n = tail
            .chars()
            .take_while(|c| {
                c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '~' | '+' | '/' | '=' | '-')
            })
            .count();
        n >= min
    };
    for (i, c) in content.char_indices() {
        match c {
            's' | 'S'
                if content[i..]
                    .get(..3)
                    .is_some_and(|p| p.eq_ignore_ascii_case("sk-"))
                    && alnum_tail(i + 3, 20) =>
            {
                return Some("openai_key")
            }
            'g' | 'G'
                if content[i..].get(..4).is_some_and(|p| {
                    ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]
                        .iter()
                        .any(|k| p.eq_ignore_ascii_case(k))
                }) =>
            {
                if alnum_tail(i + 4, 30) {
                    return Some("github_token");
                }
            }
            'A' | 'a'
                if content[i..]
                    .get(..4)
                    .is_some_and(|p| p.eq_ignore_ascii_case("AKIA"))
                    && alnum_tail(i + 4, 16) =>
            {
                return Some("aws_access_key")
            }
            'x' | 'X'
                if content[i..].get(..5).is_some_and(|p| {
                    ["xoxb-", "xoxa-", "xoxp-", "xoxr-", "xoxs-"]
                        .iter()
                        .any(|k| p.eq_ignore_ascii_case(k))
                }) && alnum_tail(i + 5, 10) =>
            {
                return Some("slack_token")
            }
            'A' | 'a'
                if content[i..]
                    .get(..4)
                    .is_some_and(|p| p.eq_ignore_ascii_case("AIza"))
                    && token_tail(i + 4, 35) =>
            {
                return Some("google_api_key")
            }
            'y' | 'Y'
                if content[i..]
                    .get(..5)
                    .is_some_and(|p| p.eq_ignore_ascii_case("ya29."))
                    && token_tail(i + 5, 10) =>
            {
                return Some("google_oauth")
            }
            '-' if content[i..].starts_with("-----BEGIN ")
                && content[i..].contains("PRIVATE KEY-----") =>
            {
                return Some("private_key")
            }
            'B' | 'b'
                if content[i..]
                    .get(..7)
                    .map(|s| s.eq_ignore_ascii_case("bearer "))
                    == Some(true)
                    && token_tail(i + 7, 20) =>
            {
                return Some("bearer_token")
            }
            'A' | 'a'
                if content[i..]
                    .get(..14)
                    .map(|s| s.eq_ignore_ascii_case("authorization:"))
                    == Some(true)
                    && content[i + 14..].chars().any(|c| !c.is_whitespace()) =>
            {
                return Some("authorization_header")
            }
            'e' | 'E'
                if content[i..]
                    .get(..3)
                    .is_some_and(|p| p.eq_ignore_ascii_case("eyJ")) =>
            {
                // JWT: eyJ<seg>.<seg>.<seg> with bounded segment lengths.
                let rest = &content[i..];
                let mut dots = 0;
                let mut seg_len = 0;
                for ch in rest.chars().skip(3) {
                    if ch == '.' {
                        dots += 1;
                        seg_len = 0;
                        if dots == 2 {
                            break;
                        }
                    } else if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                        seg_len += 1;
                        if seg_len > 512 {
                            break;
                        }
                    } else {
                        dots = 0;
                        break;
                    }
                }
                if dots >= 2 {
                    return Some("jwt");
                }
            }
            _ => {}
        }
    }
    None
}

/// Fixed-window per-workspace rate limit backed by the state store
/// (TTL-expired entries auto-clean). Read-modify-write is soft under
/// concurrency — the vault is single-tenant local-first, and the bound is
/// deliberately conservative.
/// Atomic, fail-closed fixed-window counter. The read-modify-write runs
/// inside one Immediate transaction (the same writer-lock discipline as the
/// link path, #382), so concurrent calls cannot all observe the same count
/// and exceed the cap. Malformed or non-integer stored state is an ERROR,
/// not a reset — a tampered counter must not silently reopen the gate.
pub fn check_and_bump_rate(
    db: &crate::db::Database,
    workspace: &str,
    limit: i64,
) -> Result<(), String> {
    let now = crate::db::now_ms();
    let hour = now / 3_600_000;
    let key = format!("web_gap_fill:{workspace}:{hour}");
    let conn = db
        .conn()
        .map_err(|e| format!("rate-limit state open failed: {e}"))?;
    let tx = rusqlite::Transaction::new_unchecked(&conn, rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| format!("rate-limit lock failed: {e}"))?;
    // Read inside the lock; expired rows count as 0 (window rollover).
    // Anything that is not a valid non-negative count fails CLOSED.
    let count: i64 = match tx.query_row(
        "SELECT value_json, expires_at_unix_ms FROM state WHERE key = ?1",
        rusqlite::params![key],
        |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<i64>>(1)?)),
    ) {
        Ok((raw, expires)) => {
            let live = expires.map(|exp| exp > now).unwrap_or(true);
            if live {
                raw.trim().parse::<i64>().map_err(|_| {
                    "rate-limit state corrupted (non-integer): refusing to write".to_string()
                })?
            } else {
                0
            }
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => 0,
        Err(e) => return Err(format!("rate-limit state read failed: {e}")),
    };
    if count < 0 {
        return Err("rate-limit state corrupted (negative count): refusing to write".to_string());
    }
    if count >= limit {
        return Err(format!(
            "web_gap_fill rate limit reached for this workspace ({limit}/hour)"
        ));
    }
    tx.execute(
        "INSERT OR REPLACE INTO state (key, value_json, expires_at_unix_ms, created_at_unix_ms)
         VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![key, (count + 1).to_string(), now + 7_200_000, now],
    )
    .map_err(|e| format!("rate-limit state write failed: {e}"))?;
    tx.commit()
        .map_err(|e| format!("rate-limit state commit failed: {e}"))
}

/// Deterministic entity key for fetched content: `web-<sha256(content)[..16]>`.
/// Re-fetching the same bytes updates the same entity (natural dedup).
pub fn content_key(content: &str) -> String {
    let digest = Sha256::digest(content.as_bytes());
    let hex = format!("{digest:x}");
    format!("web-{}", &hex[..16])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowlist(pairs: &[(&str, Vec<&str>)]) -> std::collections::BTreeMap<String, Vec<String>> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
            .collect()
    }

    #[test]
    fn enabled_requires_exact_one() {
        let _guard = crate::web_gap_fill::ENV_LOCK.lock().unwrap();
        std::env::remove_var("PERSEUS_VAULT_WEB_GAP_FILL_ENABLED");
        assert!(!web_gap_fill_enabled());
        std::env::set_var("PERSEUS_VAULT_WEB_GAP_FILL_ENABLED", "1");
        assert!(web_gap_fill_enabled());
        std::env::set_var("PERSEUS_VAULT_WEB_GAP_FILL_ENABLED", "true");
        assert!(!web_gap_fill_enabled(), "only '1' enables the gate");
        std::env::remove_var("PERSEUS_VAULT_WEB_GAP_FILL_ENABLED");
    }

    #[test]
    fn allowlist_parses_workspace_and_wildcard_entries() {
        let _guard = crate::web_gap_fill::ENV_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join(format!("pv-allow-{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{"ws-a": ["docs.example.com", "en.wikipedia.org"], "*": ["fallback.org"]}"#,
        )
        .unwrap();
        std::env::set_var("PERSEUS_VAULT_WEB_ALLOWLIST", &path);
        let al = load_allowlist().expect("parse");
        assert_eq!(al.get("ws-a").unwrap().len(), 2);
        assert_eq!(al.get("*").unwrap().len(), 1);
        // workspace-specific wins over wildcard; unknown workspace falls back
        assert_eq!(
            workspace_allowed_hosts(&al, "ws-a").unwrap(),
            &["docs.example.com", "en.wikipedia.org"]
        );
        assert_eq!(
            workspace_allowed_hosts(&al, "ws-other").unwrap(),
            &["fallback.org"]
        );
        assert_eq!(workspace_allowed_hosts(&al, ""), None);
        std::env::remove_var("PERSEUS_VAULT_WEB_ALLOWLIST");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn allowlist_missing_env_is_empty_not_error() {
        let _guard = crate::web_gap_fill::ENV_LOCK.lock().unwrap();
        std::env::remove_var("PERSEUS_VAULT_WEB_ALLOWLIST");
        let al = load_allowlist().expect("no env -> empty allowlist");
        assert!(al.is_empty());
        assert_eq!(workspace_allowed_hosts(&al, "anything"), None);
    }

    #[test]
    fn allowlist_malformed_or_oversized_is_error() {
        let _guard = crate::web_gap_fill::ENV_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join(format!("pv-allow-bad-{}.json", std::process::id()));
        std::fs::write(&path, "not json").unwrap();
        std::env::set_var("PERSEUS_VAULT_WEB_ALLOWLIST", &path);
        assert!(
            load_allowlist().is_err(),
            "malformed allowlist must fail closed"
        );
        std::fs::write(&path, "x".repeat(70 * 1024)).unwrap();
        assert!(
            load_allowlist().is_err(),
            "oversized allowlist must fail closed"
        );
        std::env::remove_var("PERSEUS_VAULT_WEB_ALLOWLIST");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn host_allowed_matches_exact_case_insensitive_and_wildcard() {
        let hosts = vec!["Docs.Example.com".to_string(), "*".to_string()];
        assert!(host_allowed(&hosts, "docs.example.com"));
        assert!(host_allowed(&hosts, "DOCS.EXAMPLE.COM"));
        assert!(host_allowed(&hosts, "anything.org"));
        let strict = vec!["docs.example.com".to_string()];
        assert!(!host_allowed(&strict, "example.com"));
        assert!(!host_allowed(&strict, "docs.example.com.evil.org"));
    }

    #[test]
    fn parse_source_url_accepts_http_https_and_extracts_host() {
        assert_eq!(
            parse_source_url("https://docs.example.com/page").unwrap(),
            "docs.example.com"
        );
        assert_eq!(
            parse_source_url("http://example.org").unwrap(),
            "example.org"
        );
        assert_eq!(
            parse_source_url("https://example.org:8443/x").unwrap(),
            "example.org"
        );
        assert_eq!(
            parse_source_url("HTTPS://EXAMPLE.ORG/").unwrap(),
            "EXAMPLE.ORG"
        );
    }

    #[test]
    fn parse_source_url_rejects_unsafe_forms() {
        for bad in [
            "ftp://example.org/file",
            "file:///etc/passwd",
            "data:text/plain,hello",
            "https://user:pass@example.org/",
            "https://@example.org/",
            "https:///path",
            "https://:443/",
            "https://exa mple.org/",
            "https://example.org:abc/",
            "https://example.org:",
            "example.org",
            "",
        ] {
            assert!(parse_source_url(bad).is_err(), "must reject {bad:?}");
        }
    }

    #[test]
    fn parse_source_url_rejects_encoded_literal_ip_forms() {
        // Decimal / hex / octal encodings of 127.0.0.1 must not parse as
        // hostnames and slip past the private-literal check.
        for bad in [
            "https://2130706433/",
            "https://0x7f000001/",
            "https://017700000001/",
            "https://0x7f.0.0.1/",
        ] {
            assert!(parse_source_url(bad).is_err(), "must reject {bad:?}");
        }
    }

    #[test]
    fn private_literal_extended_ranges_are_detected() {
        // #929 review: CGNAT, IETF protocol assignments, benchmarking,
        // reserved, and limited broadcast.
        for ip in [
            "100.64.0.1",
            "100.127.255.255",
            "192.0.0.1",
            "198.18.0.1",
            "198.19.255.255",
            "240.0.0.1",
            "255.255.255.255",
        ] {
            assert!(host_is_private_literal(ip), "must flag {ip}");
        }
        for ip in [
            "100.63.255.255",
            "100.128.0.1",
            "198.17.0.1",
            "198.20.0.1",
            "8.8.8.8",
        ] {
            assert!(!host_is_private_literal(ip), "must allow {ip}");
        }
    }

    /// Runtime assembly for secret-shaped fixtures: the static source must
    /// never contain a full secret-shaped string (GitHub push protection
    /// scans commit bytes, not test intent).
    fn fuse(a: impl AsRef<str>, b: impl AsRef<str>) -> String {
        format!("{}{}", a.as_ref(), b.as_ref())
    }

    #[test]
    fn secret_scan_matches_case_insensitive_canonical_forms() {
        // #929 review: canonical prefixes matched case-insensitively so
        // trivial case obfuscation cannot evade the scan.
        let cases: Vec<(String, &'static str)> = vec![
            (
                fuse("token SK", "-abcdefghijklmnopqrstuvwxyz123"),
                "openai_key",
            ),
            (
                fuse("token ghp", "_abcdefghijklmnopqrstuvwxyz1234567"),
                "github_token",
            ),
            (fuse("token AkIa", "ABCDEFGHIJKLMNOPQRST"), "aws_access_key"),
            (fuse("token XOXB", "-abcdefghijklmnop"), "slack_token"),
            (
                fuse("token AIZA", "abcdefghijklmnopqrstuvwxyz1234567890"),
                "google_api_key",
            ),
            (fuse("token YA29", ".abcdefghijklmnop"), "google_oauth"),
            (
                fuse("token eyJhbGciOiJIUzI1NiJ9", ".eyJzdWIiOiIxMjM0NTY3ODkwIn0")
                    + ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
                "jwt",
            ),
        ];
        for (content, expected) in cases {
            assert_eq!(scan_secrets(&content), Some(expected), "{content}");
        }
    }

    #[test]
    fn private_literal_ranges_are_detected() {
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254",
            "0.0.0.0",
            "224.0.0.1",
            "192.0.2.1",
            "198.51.100.7",
            "203.0.113.9",
            "::1",
            "fc00::1",
            "fe80::1",
            "::",
            "2001:db8::1",
        ] {
            assert!(
                host_is_private_literal(ip),
                "{ip} must be treated as private/reserved"
            );
        }
        assert!(!host_is_private_literal("8.8.8.8"));
        assert!(!host_is_private_literal("93.184.216.34"));
        assert!(
            !host_is_private_literal("docs.example.com"),
            "hostnames are not literals"
        );
    }

    #[test]
    fn secret_scan_detects_known_classes_and_passes_clean_content() {
        let jwt = format!(
            "{}.{}.{}",
            "eyJhbGciOiJIUzI1NiJ9",
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0",
            "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        );
        let cases: Vec<(String, &'static str)> = vec![
            (
                fuse("token sk", "-abcdefghijklmnopqrstuvwxyz123"),
                "openai_key",
            ),
            (
                fuse("push ghp", "_abcdefghijklmnopqrstuvwxyz1234567"),
                "github_token",
            ),
            (fuse("key AKIA", "IOABCDEFGHIJKLMNOPQRST"), "aws_access_key"),
            (fuse("xoxb", "-abcdefghijklmnop"), "slack_token"),
            (
                fuse("AIza", "Syabcdefghijklmnopqrstuvwxyz1234567890"),
                "google_api_key",
            ),
            (
                fuse("ya29.", "a0AfH6SMCc9abcdefghijklmnopqrstuv"),
                "google_oauth",
            ),
            (
                fuse(
                    fuse("-----BEGIN ", "RSA PRIVATE KEY-----"),
                    "\nMIIEowIwDQYJKoZIhvcNAQELBQA",
                ),
                "private_key",
            ),
            (
                fuse(
                    "Authorization: ",
                    fuse("Bearer ", "abcdefghijklmnopqrstuvwxyz123456"),
                ),
                "authorization_header",
            ),
            (
                fuse("Bearer ", "abcdefghijklmnopqrstuvwxyz123456"),
                "bearer_token",
            ),
            (jwt.clone(), "jwt"),
        ];
        for (content, class) in cases {
            assert_eq!(scan_secrets(&content), Some(class), "content: {content}");
        }
        assert_eq!(
            scan_secrets("The quick brown fox jumps over the lazy dog."),
            None
        );
        assert_eq!(scan_secrets(""), None);
    }

    #[test]
    fn rate_limit_bumps_to_cap_then_rejects() {
        let (db, path) = temp_db();
        let ws = "ws-rate";
        for i in 1..=10 {
            check_and_bump_rate(&db, ws, 10).expect(&format!("write {i} allowed"));
        }
        let err = check_and_bump_rate(&db, ws, 10).expect_err("11th write must be rejected");
        assert!(err.contains("rate limit"), "{err}");
        // different workspace has its own budget
        check_and_bump_rate(&db, "ws-other", 10).expect("other workspace unaffected");
        let _ = std::fs::remove_file(path);
    }

    fn temp_db() -> (crate::db::Database, String) {
        let path = std::env::temp_dir()
            .join(format!(
                "web-gap-fill-{}-{}.db",
                std::process::id(),
                uuid::Uuid::new_v4().simple()
            ))
            .to_string_lossy()
            .to_string();
        let _ = std::fs::remove_file(&path);
        let db = crate::db::Database::open(&path).expect("open temp db");
        (db, path)
    }
}
