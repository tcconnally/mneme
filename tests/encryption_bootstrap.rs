// #850: encryption-by-default bootstrap contract, exercised end-to-end through
// the real CLI with a private HOME.
//
// These tests MUST spawn the binary as a subprocess with a sandboxed HOME
// rather than mutating the test process's environment: `default_db_path()` is
// eagerly evaluated by clap for every `serve`/`write` parse, so a test that
// calls `std::env::set_var("HOME", …)` in-process races with every other test
// that parses a CLI (clap reads HOME mid-parse and `apply_top_level_db`'s
// `*db == default_db_path()` comparison then sees a different value). A child
// process owns its env, so the parent never touches global state.

use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_perseus-vault");

fn sandbox(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("perseus-enc-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn assert_no_plaintext_bytes(paths: &[&Path], sentinels: &[&str]) {
    for path in paths {
        if !path.is_file() {
            continue;
        }
        let bytes = std::fs::read(path).unwrap();
        for sentinel in sentinels {
            assert!(
                !bytes
                    .windows(sentinel.len())
                    .any(|window| window == sentinel.as_bytes()),
                "{} retained plaintext sentinel {}",
                path.display(),
                sentinel
            );
        }
    }
}

#[test]
fn fresh_default_database_creates_key_canary_and_encrypts_bodies() {
    let home = sandbox("home");
    let db_path = home.join("data").join("perseus-vault.db");
    std::fs::create_dir_all(db_path.parent().unwrap()).unwrap();

    let out = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "bootstrap-key",
            "--body-json",
            r#"{"note":"bootstrap encrypted body"}"#,
        ])
        .output()
        .expect("spawn perseus-vault write");

    assert!(
        out.status.success(),
        "write must succeed under default encryption\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Standard key path under the sandboxed HOME, owner-only on Unix.
    let key_path = home.join(".perseus-vault").join("secret.key");
    assert!(
        key_path.is_file(),
        "key file must exist at {}",
        key_path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&key_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "key file must be 0600 on Unix");
    }
    let key_material = std::fs::read_to_string(&key_path).unwrap();
    let key_material = key_material.trim();
    let decoded = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, key_material)
        .expect("key file must be valid base64");
    assert_eq!(decoded.len(), 32, "key file must hold a 32-byte key");

    // Canary established and the written body is ciphertext at rest.
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let canary: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM encryption_canary WHERE id = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(canary, 1, "encrypted canary must be established by default");
    let stored: String = conn
        .query_row(
            "SELECT body_json FROM entities WHERE key='bootstrap-key'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_ne!(stored, r#"{"note":"bootstrap encrypted body"}"#);
    assert!(
        !stored.contains("bootstrap encrypted body"),
        "ciphertext must not leak the plaintext: {stored}"
    );

    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn reindex_preserves_protected_activation_markers() {
    let home = sandbox("reindex-activation");
    let db_path = home.join("encrypted.db");

    let write = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "reindex-activation",
            "--body-json",
            r#"{"note":"reindex activation sentinel"}"#,
        ])
        .output()
        .expect("spawn encrypted reindex seed");
    assert!(
        write.status.success(),
        "encrypted seed failed: {}",
        String::from_utf8_lossy(&write.stderr)
    );

    let key_path = home.join(".perseus-vault").join("secret.key");
    let reindex = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "reindex",
            "--db",
            db_path.to_str().unwrap(),
            "--encryption-key",
            key_path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn encrypted reindex");
    assert!(
        reindex.status.success(),
        "encrypted reindex failed: {}",
        String::from_utf8_lossy(&reindex.stderr)
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    assert_eq!(
        conn.query_row::<i64, _, _>(
            "SELECT COUNT(*) FROM encryption_canary WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap(),
        1,
        "successful protected reindex must reactivate the canary"
    );
    assert_eq!(
        conn.query_row::<String, _, _>(
            "SELECT search_mode FROM encryption_profile WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap(),
        "hmac-sha256-blind-token-v1",
        "successful protected reindex must reactivate the profile"
    );

    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn encrypted_fts_does_not_persist_plaintext_body() {
    let home = sandbox("fts-confidentiality");
    let db_path = home.join("encrypted.db");
    let sentinel = "fts-confidentiality-sentinel-7f3e";
    let body = format!(r#"{{"note":"{sentinel}"}}"#);

    let out = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "fts-confidentiality",
            "--body-json",
            &body,
        ])
        .output()
        .expect("spawn encrypted FTS write");
    assert!(
        out.status.success(),
        "encrypted write failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let indexed: String = conn
        .query_row(
            "SELECT body_json FROM entities_fts WHERE rowid = (SELECT rowid FROM entities WHERE key = 'fts-confidentiality')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        !indexed.contains(sentinel),
        "encrypted FTS must not retain the body plaintext: {indexed}"
    );
    let shadow: String = conn
        .query_row(
            "SELECT c0 FROM entities_fts_content WHERE rowid = (SELECT rowid FROM entities WHERE key = 'fts-confidentiality')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        !shadow.contains(sentinel),
        "FTS shadow content must not retain the body plaintext: {shadow}"
    );

    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn explicit_plaintext_optout_suppresses_default_key_creation() {
    let home = sandbox("optout");
    let db_path = home.join("perseus-vault.db");

    let out = Command::new(BIN)
        .env("HOME", &home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "optout-key",
            "--body-json",
            r#"{"note":"explicit plaintext opt-out"}"#,
        ])
        .output()
        .expect("spawn perseus-vault write");

    assert!(
        out.status.success(),
        "write must succeed under explicit plaintext opt-out\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let key_path = home.join(".perseus-vault").join("secret.key");
    assert!(
        !key_path.exists(),
        "no key file may be created under explicit opt-out: {}",
        key_path.display()
    );

    // Body is stored as plaintext (no encryption was applied).
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let stored: String = conn
        .query_row(
            "SELECT body_json FROM entities WHERE key='optout-key'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stored, r#"{"note":"explicit plaintext opt-out"}"#);

    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn init_rekey_migrates_existing_plaintext_rows_and_is_idempotent() {
    let home = sandbox("rekey-migration");
    let db_path = home.join("legacy-plaintext.db");
    let key_path = home.join("migration-key");
    let plaintext = r#"{"note":"legacy plaintext must be migrated"}"#;

    // Establish the pre-migration state explicitly: a real plaintext body and
    // no encryption canary, using the documented compatibility opt-out.
    let write = Command::new(BIN)
        .env("HOME", &home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "legacy-plaintext",
            "--body-json",
            plaintext,
        ])
        .output()
        .expect("spawn plaintext seed write");
    assert!(
        write.status.success(),
        "plaintext seed must succeed: {}",
        String::from_utf8_lossy(&write.stderr)
    );
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    assert_eq!(
        conn.query_row::<i64, _, _>(
            "SELECT COUNT(*) FROM encryption_canary WHERE id = 1",
            [],
            |r| r.get(0)
        )
        .unwrap(),
        0,
        "the fixture must begin plaintext"
    );
    assert_eq!(
        conn.query_row::<String, _, _>(
            "SELECT body_json FROM entities WHERE key='legacy-plaintext'",
            [],
            |r| r.get(0)
        )
        .unwrap(),
        plaintext
    );
    drop(conn);

    // The explicit migration establishes the canary and encrypts the existing
    // body under the operator-provided key.
    let migrate = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "init",
            "--db",
            db_path.to_str().unwrap(),
            "--key-file",
            key_path.to_str().unwrap(),
            "--rekey",
        ])
        .output()
        .expect("spawn init --rekey");
    let migrate_stdout = String::from_utf8_lossy(&migrate.stdout);
    assert!(
        migrate.status.success(),
        "init --rekey must succeed: {migrate_stdout}\n{}",
        String::from_utf8_lossy(&migrate.stderr)
    );
    assert!(
        migrate_stdout.contains("encrypt: 1 records encrypted, 0 skipped, 0 failed"),
        "migration report must account for the legacy row: {migrate_stdout}"
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    assert_eq!(
        conn.query_row::<i64, _, _>(
            "SELECT COUNT(*) FROM encryption_canary WHERE id = 1",
            [],
            |r| r.get(0)
        )
        .unwrap(),
        1,
        "rekey must establish the encryption canary"
    );
    let stored: String = conn
        .query_row(
            "SELECT body_json FROM entities WHERE key='legacy-plaintext'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_ne!(stored, plaintext, "legacy body must no longer be plaintext");
    assert!(
        !stored.contains("legacy plaintext must be migrated"),
        "ciphertext must not contain the migrated body: {stored}"
    );
    drop(conn);

    // A second explicit migration is safe and must not double-encrypt or
    // replace the key/ciphertext.
    let migrate_again = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "init",
            "--db",
            db_path.to_str().unwrap(),
            "--key-file",
            key_path.to_str().unwrap(),
            "--rekey",
        ])
        .output()
        .expect("spawn idempotent init --rekey");
    let again_stdout = String::from_utf8_lossy(&migrate_again.stdout);
    assert!(
        migrate_again.status.success(),
        "idempotent init --rekey must succeed: {again_stdout}\n{}",
        String::from_utf8_lossy(&migrate_again.stderr)
    );
    assert!(
        again_stdout.contains("encrypt: 0 records encrypted, 1 skipped, 0 failed"),
        "second migration must skip the already encrypted row: {again_stdout}"
    );

    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn unkeyed_migrate_rejects_encrypted_incomplete_target() {
    let home = sandbox("migrate-incomplete-target");
    let target = home.join("target.db");
    let source = home.join("source.db");
    let key_path = home.join("target.key");

    let keygen = Command::new(BIN)
        .env("HOME", &home)
        .args(["keygen", "--key-file", key_path.to_str().unwrap()])
        .output()
        .expect("spawn keygen");
    assert!(
        keygen.status.success(),
        "keygen failed: {}",
        String::from_utf8_lossy(&keygen.stderr)
    );
    let write = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "write",
            "--db",
            target.to_str().unwrap(),
            "--encryption-key",
            key_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "incomplete-target",
            "--body-json",
            r#"{"note":"incomplete-target-sentinel"}"#,
        ])
        .output()
        .expect("spawn encrypted target write");
    assert!(
        write.status.success(),
        "encrypted target seed failed: {}",
        String::from_utf8_lossy(&write.stderr)
    );

    let conn = rusqlite::Connection::open(&target).unwrap();
    conn.execute("DELETE FROM entities_fts", []).unwrap();
    drop(conn);
    std::fs::write(&source, b"").unwrap();

    let migrate = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "migrate",
            "--from",
            source.to_str().unwrap(),
            "--to",
            target.to_str().unwrap(),
        ])
        .output()
        .expect("spawn unkeyed migrate");
    assert!(
        !migrate.status.success(),
        "unkeyed migration must reject an incomplete encrypted target: stdout={} stderr={}",
        String::from_utf8_lossy(&migrate.stdout),
        String::from_utf8_lossy(&migrate.stderr)
    );
    assert!(
        String::from_utf8_lossy(&migrate.stderr).contains("encrypted"),
        "rejection must identify the target as encrypted/incomplete: {}",
        String::from_utf8_lossy(&migrate.stderr)
    );

    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn init_rekey_migrates_archived_plaintext_rows_and_preserves_archive_state() {
    let home = sandbox("rekey-archived");
    let db_path = home.join("legacy-archived.db");
    let key_path = home.join("migration-key");
    let plaintext = r#"{"note":"archived plaintext must be migrated"}"#;

    let write = Command::new(BIN)
        .env("HOME", &home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "archived-plaintext",
            "--body-json",
            plaintext,
        ])
        .output()
        .expect("spawn archived plaintext seed write");
    assert!(
        write.status.success(),
        "archived plaintext seed must succeed: {}",
        String::from_utf8_lossy(&write.stderr)
    );

    let forget = Command::new(BIN)
        .env("HOME", &home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
        .args([
            "forget",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "archived-plaintext",
            "--reason",
            "test archive",
        ])
        .output()
        .expect("spawn archive command");
    assert!(
        forget.status.success(),
        "archive command must succeed: {}",
        String::from_utf8_lossy(&forget.stderr)
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let (entity_id, archived_before, stored_before): (String, i64, String) = conn
        .query_row(
            "SELECT id, archived, body_json FROM entities WHERE key='archived-plaintext'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(archived_before, 1, "fixture must archive the entity");
    assert!(
        stored_before == plaintext,
        "fixture must begin with the expected plaintext body"
    );
    let (signature_before, _signature_len_before): (i64, i64) = conn
        .query_row(
            "SELECT body_hash, body_len FROM dedup_signatures WHERE entity_id = ?1",
            rusqlite::params![entity_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    drop(conn);

    let migrate = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "init",
            "--db",
            db_path.to_str().unwrap(),
            "--key-file",
            key_path.to_str().unwrap(),
            "--rekey",
        ])
        .output()
        .expect("spawn archived init --rekey");
    let migrate_stdout = String::from_utf8_lossy(&migrate.stdout);
    assert!(
        migrate.status.success(),
        "archived init --rekey must succeed: {}",
        String::from_utf8_lossy(&migrate.stderr)
    );
    assert!(
        migrate_stdout.contains("encrypt: 1 records encrypted, 0 skipped, 0 failed"),
        "archived row must be included in migration counts: {migrate_stdout}"
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let (archived_after, stored_after): (i64, String) = conn
        .query_row(
            "SELECT archived, body_json FROM entities WHERE key='archived-plaintext'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(archived_after, 1, "rekey must preserve archive state");
    assert!(
        stored_after != plaintext,
        "archived body must no longer be plaintext"
    );
    assert!(
        !stored_after.contains("archived plaintext must be migrated"),
        "ciphertext must not contain archived body content"
    );
    let (signature_after, signature_len_after): (i64, i64) = conn
        .query_row(
            "SELECT body_hash, body_len FROM dedup_signatures WHERE entity_id = ?1",
            rusqlite::params![entity_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert!(
        signature_after != signature_before,
        "rekey must refresh the dedup signature"
    );
    assert!(
        signature_len_after == stored_after.len() as i64,
        "dedup signature length must match the stored ciphertext"
    );
    drop(conn);

    let verify = Command::new(BIN)
        .env("HOME", &home)
        .args(["verify", "--db", db_path.to_str().unwrap(), "--json"])
        .output()
        .expect("spawn encrypted-state verification");
    let report: serde_json::Value =
        serde_json::from_slice(&verify.stdout).expect("verify --json must emit JSON");
    let c2 = report["checks"]
        .as_array()
        .and_then(|checks| checks.iter().find(|check| check["id"] == "C2"))
        .expect("verify report must include C2");
    assert_eq!(c2["status"], "PASS", "C2 must pass after archived rekey");
    assert_eq!(c2["findings"], serde_json::json!([]));

    let migrate_again = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "init",
            "--db",
            db_path.to_str().unwrap(),
            "--key-file",
            key_path.to_str().unwrap(),
            "--rekey",
        ])
        .output()
        .expect("spawn idempotent archived init --rekey");
    let again_stdout = String::from_utf8_lossy(&migrate_again.stdout);
    assert!(
        migrate_again.status.success(),
        "second archived init --rekey must succeed: {}",
        String::from_utf8_lossy(&migrate_again.stderr)
    );
    assert!(
        again_stdout.contains("encrypt: 0 records encrypted, 1 skipped, 0 failed"),
        "second archived migration must be a no-op: {again_stdout}"
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let (archived_final, stored_final): (i64, String) = conn
        .query_row(
            "SELECT archived, body_json FROM entities WHERE key='archived-plaintext'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        archived_final, 1,
        "idempotent rekey must preserve archive state"
    );
    assert!(
        stored_final == stored_after,
        "idempotent rekey must preserve ciphertext"
    );
    drop(conn);

    let _ = std::fs::remove_dir_all(&home);
}

// #1018 companion guard: keygen/init previously truncated any key file at the
// resolved path, which would destroy the key of an existing encrypted vault
// (precedence resolution now finds legacy `~/.mimir/secret.key` files, making
// the footgun reachable again). Keygen must fail closed on an existing key;
// init must USE the existing key as-is ("generates a key, if none exists")
// and leave its bytes untouched.
#[test]
fn keygen_refuses_and_init_reuses_an_existing_key_file() {
    let home = sandbox("keyguard");
    let key_path = home.join("secret.key");
    let db_path = home.join("vault.db");

    // Create a real key with keygen (fresh path).
    let keygen = Command::new(BIN)
        .env("HOME", &home)
        .args(["keygen", "--key-file", key_path.to_str().unwrap()])
        .output()
        .expect("spawn perseus-vault keygen");
    assert!(
        keygen.status.success(),
        "keygen on a fresh path must succeed: {}",
        String::from_utf8_lossy(&keygen.stderr)
    );
    let key_before = std::fs::read_to_string(&key_path).unwrap();

    // keygen again on the same path: refused, key untouched.
    let keygen2 = Command::new(BIN)
        .env("HOME", &home)
        .args(["keygen", "--key-file", key_path.to_str().unwrap()])
        .output()
        .expect("spawn perseus-vault keygen");
    assert!(
        !keygen2.status.success(),
        "keygen must refuse to overwrite an existing key file"
    );
    assert!(
        String::from_utf8_lossy(&keygen2.stderr)
            .contains("refusing to overwrite existing key file"),
        "refusal must name the action: {}",
        String::from_utf8_lossy(&keygen2.stderr)
    );
    assert_eq!(std::fs::read_to_string(&key_path).unwrap(), key_before);

    // init on the same key: reuses it, succeeds, bytes untouched.
    let init = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "init",
            "--db",
            db_path.to_str().unwrap(),
            "--key-file",
            key_path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn perseus-vault init");
    assert!(
        init.status.success(),
        "init must reuse an existing key: {}",
        String::from_utf8_lossy(&init.stderr)
    );
    assert_eq!(std::fs::read_to_string(&key_path).unwrap(), key_before);
    // The database was encrypted with that key (canary established).
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let canary: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM encryption_canary WHERE id = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        canary, 1,
        "init must establish the canary with the existing key"
    );
    drop(conn);

    let _ = std::fs::remove_dir_all(&home);
}

// #1018: a v2.21-era encrypted vault (canary under the pre-rebrand
// `mimir_internal` AAD, key at the legacy `~/.mimir/secret.key` path) must
// open with the CURRENT binary using that same key — no key regeneration, no
// manual DB repair — and recall must keep working through the MCP server.
// The fixture is synthesized by re-encrypting the canary under the legacy AAD
// with the same AES-256-GCM scheme the binary uses (aes-gcm is a direct
// dependency, so the wire format matches by construction).
#[test]
fn legacy_mimir_vault_opens_and_recalls_with_current_binary() {
    use aes_gcm::aead::rand_core::RngCore;
    use aes_gcm::aead::{Aead, KeyInit, OsRng};
    use aes_gcm::{Aes256Gcm, Key, Nonce};
    use base64::Engine as _;
    use std::io::{BufRead, BufReader, Write};
    use std::process::Stdio;

    let home = sandbox("legacy1018");
    let db_path = home.join("mimir.db");
    let key_path = home.join(".mimir").join("secret.key");
    std::fs::create_dir_all(key_path.parent().unwrap()).unwrap();

    // 1. v2.21-era key at the legacy path.
    let keygen = Command::new(BIN)
        .env("HOME", &home)
        .args(["keygen", "--key-file", key_path.to_str().unwrap()])
        .output()
        .expect("spawn perseus-vault keygen");
    assert!(
        keygen.status.success(),
        "keygen failed: {}",
        String::from_utf8_lossy(&keygen.stderr)
    );

    // 2. Bootstrap the encrypted fixture with the current binary.
    let write1 = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--encryption-key",
            key_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "legacy-entity",
            "--body-json",
            r#"{"note":"pre-rebrand secret"}"#,
        ])
        .output()
        .expect("spawn perseus-vault write");
    assert!(
        write1.status.success(),
        "fixture write failed: {}",
        String::from_utf8_lossy(&write1.stderr)
    );

    // 3. Downgrade the canary to the v2.21-era AAD (`mimir_internal`).
    let key_b64 = std::fs::read_to_string(&key_path).unwrap();
    let key_bytes = base64::engine::general_purpose::STANDARD
        .decode(key_b64.trim())
        .unwrap();
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key_bytes));
    let current_aad = format!(
        "{}:{}:{}:{}",
        "perseus_vault_internal".len(),
        "perseus_vault_internal",
        "encryption_canary".len(),
        "encryption_canary"
    );
    let legacy_aad = format!(
        "{}:{}:{}",
        "mimir_internal".len(),
        "mimir_internal",
        "encryption_canary"
    );
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let stored: String = conn
        .query_row(
            "SELECT ciphertext FROM encryption_canary WHERE id = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let combined = base64::engine::general_purpose::STANDARD
        .decode(&stored)
        .unwrap();
    let (nonce_bytes, ct_body) = combined.split_at(12);
    let plain = cipher
        .decrypt(
            Nonce::from_slice(nonce_bytes),
            aes_gcm::aead::Payload {
                msg: ct_body,
                aad: current_aad.as_bytes(),
            },
        )
        .expect("fixture canary must decrypt under the current AAD");
    let mut new_nonce = [0u8; 12];
    OsRng.fill_bytes(&mut new_nonce);
    let legacy_ct = cipher
        .encrypt(
            Nonce::from_slice(&new_nonce),
            aes_gcm::aead::Payload {
                msg: plain.as_slice(),
                aad: legacy_aad.as_bytes(),
            },
        )
        .expect("re-encrypt canary under the legacy AAD");
    // App format is base64(nonce[12] || ciphertext) — prepend the nonce.
    let mut combined = new_nonce.to_vec();
    combined.extend_from_slice(&legacy_ct);
    let legacy_ct_b64 = base64::engine::general_purpose::STANDARD.encode(combined);
    conn.execute(
        "UPDATE encryption_canary SET ciphertext = ?1 WHERE id = 1",
        rusqlite::params![legacy_ct_b64],
    )
    .unwrap();
    drop(conn);

    // 4. The CURRENT binary must open the v2.21-era vault with its existing
    //    key. Pre-fix this exits 1 with "failed to decrypt encryption canary".
    let write2 = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--encryption-key",
            key_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "legacy-entity-2",
            "--body-json",
            r#"{"note":"post-upgrade secret"}"#,
        ])
        .output()
        .expect("spawn perseus-vault write");
    let stderr2 = String::from_utf8_lossy(&write2.stderr);
    assert!(
        write2.status.success(),
        "upgraded binary must open a v2.21-era encrypted vault with its existing key\nstderr: {stderr2}"
    );
    assert!(
        !stderr2.contains("failed to decrypt encryption canary"),
        "the canary regression error must not fire: {stderr2}"
    );

    // 5. Startup + recall through the real MCP stdio server, same key.
    let mut child = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "serve",
            "--db",
            db_path.to_str().unwrap(),
            "--encryption-key",
            key_path.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn perseus-vault serve");
    let mut stdin = child.stdin.take().unwrap();
    let stdout_handle = child.stdout.take().unwrap();

    // Reader thread: the stdio server answers on stdout. A channel decouples
    // reading from the bounded wait below, so a silent server can never hang
    // the CI job forever (the pre-fix #1018 crash is one failure mode; a
    // stuck server is the other).
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        let stdout = BufReader::new(stdout_handle);
        for line in stdout.lines() {
            match line {
                Ok(l) => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    // One JSON-RPC exchange: initialize, then recall. Fails fast if the
    // server exits (the #1018 regression) or goes silent (hard 240s
    // deadline per call, then the child is killed).
    fn mcp_call(
        stdin: &mut std::process::ChildStdin,
        child: &mut std::process::Child,
        rx: &std::sync::mpsc::Receiver<String>,
        payload: &str,
        target: u64,
    ) -> String {
        use std::time::{Duration, Instant};
        writeln!(stdin, "{payload}").unwrap();
        let deadline = Duration::from_secs(240);
        let start = Instant::now();
        loop {
            match rx.recv_timeout(Duration::from_secs(5)) {
                Ok(line) => {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                        if v.get("id").and_then(|i| i.as_u64()) == Some(target) {
                            return line;
                        }
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if let Ok(Some(status)) = child.try_wait() {
                        panic!("serve exited ({status}) before answering id {target}");
                    }
                    if start.elapsed() >= deadline {
                        let _ = child.kill();
                        panic!("serve did not answer id {target} within {deadline:?} — killed it");
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    panic!("serve stdout closed before answering id {target}");
                }
            }
        }
    }

    let init = mcp_call(
        &mut stdin,
        &mut child,
        &rx,
        r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}"#,
        0,
    );
    assert!(init.contains("\"result\""), "initialize failed: {init}");

    // Initialized notification — no response expected.
    stdin
        .write_all(br#"{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}"#)
        .unwrap();
    stdin.write_all(b"\n").unwrap();

    let recall = mcp_call(
        &mut stdin,
        &mut child,
        &rx,
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"perseus_vault_recall","arguments":{"query":"pre-rebrand secret"}}}"#,
        1,
    );
    assert!(recall.contains("\"result\""), "recall failed: {recall}");
    assert!(
        recall.contains("legacy-entity"),
        "recall must return the pre-rebrand entity: {recall}"
    );

    // Shutdown: closing stdin makes the stdio server exit (EOF), which also
    // ends the reader thread.
    drop(stdin);
    let _ = child.wait();
    let _ = reader.join();
    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn init_encrypts_legacy_bodies_before_advertising_protected_search() {
    let home = sandbox("legacy-body-coverage");
    let db_path = home.join("legacy.db");
    let key_path = home.join("migration-key");
    let sentinel = "legacy-body-coverage-sentinel-9c4e";
    let plaintext = format!(r#"{{"note":"{sentinel}"}}"#);

    let seed = Command::new(BIN)
        .env("HOME", &home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "legacy-body",
            "--body-json",
            &plaintext,
        ])
        .output()
        .expect("spawn plaintext seed");
    assert!(
        seed.status.success(),
        "plaintext seed failed: {}",
        String::from_utf8_lossy(&seed.stderr)
    );

    let init = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "init",
            "--db",
            db_path.to_str().unwrap(),
            "--key-file",
            key_path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn init");
    assert!(
        init.status.success(),
        "init must migrate legacy bodies: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let stored: String = conn
        .query_row(
            "SELECT body_json FROM entities WHERE key='legacy-body'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_ne!(stored, plaintext, "the canonical body must be encrypted");
    assert!(
        !stored.contains(sentinel),
        "the canonical ciphertext must not expose the sentinel"
    );
    let indexed: String = conn
        .query_row(
            "SELECT body_json FROM entities_fts WHERE rowid=(SELECT rowid FROM entities WHERE key='legacy-body')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        !indexed.contains(sentinel),
        "protected FTS must not expose the legacy sentinel"
    );
    let mode: String = conn
        .query_row(
            "SELECT search_mode FROM encryption_profile WHERE id=1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(mode, "hmac-sha256-blind-token-v1");

    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn failed_encryption_migration_does_not_leave_a_valid_canary() {
    let home = sandbox("migration-failure-canary");
    let db_path = home.join("legacy.db");
    let key_path = home.join("migration.key");
    let plaintext = r#"{"note":"migration-failure-canary-sentinel-6a91"}"#;

    let seed = Command::new(BIN)
        .env("HOME", &home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "migration-failure",
            "--body-json",
            plaintext,
        ])
        .output()
        .expect("spawn plaintext migration seed");
    assert!(
        seed.status.success(),
        "plaintext seed failed: {}",
        String::from_utf8_lossy(&seed.stderr)
    );

    // Make the legacy row structurally invalid only at the protected-FTS
    // rebuild boundary. The migration must fail closed and roll back its body
    // update; the old implementation nevertheless wrote the canary first.
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute(
        "UPDATE entities SET hints = 'not-json' WHERE key = 'migration-failure'",
        [],
    )
    .unwrap();
    drop(conn);

    let init = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "init",
            "--db",
            db_path.to_str().unwrap(),
            "--key-file",
            key_path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn failing migration");
    assert!(
        !init.status.success(),
        "malformed hints must fail migration"
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let canary_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM encryption_canary WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        canary_count, 0,
        "failed migration must not leave a canary that blesses plaintext rows"
    );
    let pending_mode: String = conn
        .query_row(
            "SELECT search_mode FROM encryption_profile WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        pending_mode, "migration-pending",
        "failed migration must remain diagnosable as incomplete"
    );
    let stored: String = conn
        .query_row(
            "SELECT body_json FROM entities WHERE key = 'migration-failure'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        stored, plaintext,
        "failed migration must roll back the body write"
    );

    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn init_encrypts_history_bodies_and_history_fts() {
    let home = sandbox("legacy-history-coverage");
    let db_path = home.join("legacy-history.db");
    let key_path = home.join("migration-key");
    let first = r#"{"note":"history-coverage-first-4c1a"}"#;
    let second = r#"{"note":"history-coverage-second-4c1a"}"#;

    for body in [first, second] {
        let write = Command::new(BIN)
            .env("HOME", &home)
            .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
            .args([
                "write",
                "--db",
                db_path.to_str().unwrap(),
                "--category",
                "facts",
                "--key",
                "history-coverage",
                "--body-json",
                body,
            ])
            .output()
            .expect("spawn plaintext history seed");
        assert!(
            write.status.success(),
            "history seed failed: {}",
            String::from_utf8_lossy(&write.stderr)
        );
    }

    let init = Command::new(BIN)
        .env("HOME", &home)
        .env_remove("PERSEUS_VAULT_ALLOW_PLAINTEXT")
        .args([
            "init",
            "--db",
            db_path.to_str().unwrap(),
            "--key-file",
            key_path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn history init");
    assert!(
        init.status.success(),
        "history init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let history_bodies: Vec<String> = conn
        .prepare("SELECT body_json FROM entity_history")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(
        !history_bodies.is_empty(),
        "the second write must create a history row"
    );
    assert!(
        history_bodies
            .iter()
            .all(|body| !body.contains("history-coverage-first-4c1a")),
        "history bodies must not retain plaintext content"
    );
    let history_fts: Vec<String> = conn
        .prepare("SELECT body_json FROM entity_history_fts")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(
        history_fts
            .iter()
            .all(|body| !body.contains("history-coverage-first-4c1a")),
        "history FTS must not retain plaintext content"
    );

    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn encrypted_redaction_keeps_marker_encrypted_and_removes_history_fts() {
    let home = sandbox("encrypted-redaction");
    let db_path = home.join("redaction.db");
    let key_path = home.join("redaction.key");
    let old_body = r#"{"note":"redaction-old-sentinel-2d77"}"#;
    let new_body = r#"{"note":"redaction-new-sentinel-2d77"}"#;

    let keygen = Command::new(BIN)
        .env("HOME", &home)
        .args(["keygen", "--key-file", key_path.to_str().unwrap()])
        .output()
        .expect("spawn redaction keygen");
    assert!(
        keygen.status.success(),
        "keygen failed: {}",
        String::from_utf8_lossy(&keygen.stderr)
    );
    for body in [old_body, new_body] {
        let write = Command::new(BIN)
            .env("HOME", &home)
            .args([
                "write",
                "--db",
                db_path.to_str().unwrap(),
                "--encryption-key",
                key_path.to_str().unwrap(),
                "--category",
                "facts",
                "--key",
                "redaction-key",
                "--workspace-hash",
                "redaction-workspace",
                "--body-json",
                body,
            ])
            .output()
            .expect("spawn encrypted redaction write");
        assert!(
            write.status.success(),
            "encrypted write failed: {}",
            String::from_utf8_lossy(&write.stderr)
        );
    }

    let redact = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "redact",
            "--db",
            db_path.to_str().unwrap(),
            "--encryption-key",
            key_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "redaction-key",
            "--workspace-hash",
            "redaction-workspace",
        ])
        .output()
        .expect("spawn encrypted redaction");
    assert!(
        redact.status.success(),
        "encrypted redaction failed: {}",
        String::from_utf8_lossy(&redact.stderr)
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let stored: String = conn
        .query_row(
            "SELECT body_json FROM entities WHERE key = 'redaction-key'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        !stored.contains("\"redacted\""),
        "redaction marker must be encrypted"
    );
    assert!(!stored.contains(old_body) && !stored.contains(new_body));
    let history_fts_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM entity_history_fts", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(
        history_fts_count, 0,
        "redaction must remove history FTS rows"
    );
    drop(conn);

    let doctor = Command::new(BIN)
        .env("HOME", &home)
        .args(["doctor", "--db", db_path.to_str().unwrap()])
        .output()
        .expect("spawn encrypted redaction doctor");
    assert!(
        doctor.status.success(),
        "doctor failed after redaction: {}",
        String::from_utf8_lossy(&doctor.stderr)
    );
    assert!(
        String::from_utf8_lossy(&doctor.stdout).contains("encrypted"),
        "redaction must not downgrade the storage diagnostic: {}",
        String::from_utf8_lossy(&doctor.stdout)
    );

    let _ = std::fs::remove_dir_all(&home);
}

#[test]
fn key_rotation_reencrypts_live_history_fts_and_backup() {
    let home = sandbox("rotation");
    let db_path = home.join("vault.db");
    let old_key = home.join("old.key");
    let new_key = home.join("new.key");
    let backup_path = home.join("vault-backup.db");
    let old_body = r#"{"note":"rotation-old-live-8c2e"}"#;
    let new_body = r#"{"note":"rotation-new-live-8c2e"}"#;
    let post_rotation_body = r#"{"note":"rotation-after-write-8c2e"}"#;

    for key in [&old_key, &new_key] {
        let generated = Command::new(BIN)
            .env("HOME", &home)
            .args(["keygen", "--key-file", key.to_str().unwrap()])
            .output()
            .expect("spawn keygen");
        assert!(
            generated.status.success(),
            "keygen failed: {}",
            String::from_utf8_lossy(&generated.stderr)
        );
    }

    let first = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--encryption-key",
            old_key.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "rotation-key",
            "--body-json",
            old_body,
        ])
        .output()
        .expect("spawn first encrypted write");
    assert!(
        first.status.success(),
        "first write failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let second = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--encryption-key",
            old_key.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "rotation-key",
            "--body-json",
            new_body,
        ])
        .output()
        .expect("spawn history-producing write");
    assert!(
        second.status.success(),
        "second write failed: {}",
        String::from_utf8_lossy(&second.stderr)
    );

    let rotated = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "rotate-key",
            "--db",
            db_path.to_str().unwrap(),
            "--old-key",
            old_key.to_str().unwrap(),
            "--new-key",
            new_key.to_str().unwrap(),
        ])
        .output()
        .expect("spawn key rotation");
    assert!(
        rotated.status.success(),
        "key rotation failed: {}",
        String::from_utf8_lossy(&rotated.stderr)
    );

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    for table in [
        "entities",
        "entity_history",
        "entities_fts",
        "entity_history_fts",
    ] {
        let query = format!("SELECT body_json FROM {table}");
        let values: Vec<String> = conn
            .prepare(&query)
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(
            values
                .iter()
                .all(|value| !value.contains(old_body) && !value.contains(new_body)),
            "{table} retained rotated plaintext"
        );
    }
    drop(conn);
    assert_no_plaintext_bytes(&[db_path.as_path()], &[old_body, new_body]);

    let new_write = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--encryption-key",
            new_key.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "rotation-after-write",
            "--body-json",
            post_rotation_body,
        ])
        .output()
        .expect("spawn post-rotation write");
    assert!(
        new_write.status.success(),
        "new key could not open rotated store: {}",
        String::from_utf8_lossy(&new_write.stderr)
    );

    let old_write = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--encryption-key",
            old_key.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "old-key-must-fail",
            "--body-json",
            r#"{"note":"old-key-must-not-write"}"#,
        ])
        .output()
        .expect("spawn wrong-key write");
    assert!(
        !old_write.status.success(),
        "old key must fail after rotation"
    );

    let backup = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "backup",
            "--db",
            db_path.to_str().unwrap(),
            "--to",
            backup_path.to_str().unwrap(),
            "--encryption-key",
            new_key.to_str().unwrap(),
        ])
        .output()
        .expect("spawn encrypted backup");
    assert!(
        backup.status.success(),
        "encrypted backup failed: {}",
        String::from_utf8_lossy(&backup.stderr)
    );
    assert!(backup_path.is_file(), "backup destination must be created");

    let restored_path = home.join("vault-restored.db");
    let restored = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "restore",
            "--from",
            backup_path.to_str().unwrap(),
            "--to",
            restored_path.to_str().unwrap(),
            "--encryption-key",
            new_key.to_str().unwrap(),
        ])
        .output()
        .expect("spawn encrypted restore");
    assert!(
        restored.status.success(),
        "encrypted restore failed: {}",
        String::from_utf8_lossy(&restored.stderr)
    );
    assert!(
        restored_path.is_file(),
        "restore destination must be created"
    );
    let restored_conn = rusqlite::Connection::open(&restored_path).unwrap();
    let restored_check: String = restored_conn
        .query_row("PRAGMA quick_check", [], |row| row.get(0))
        .unwrap();
    assert_eq!(
        restored_check, "ok",
        "restored database must pass quick_check"
    );
    let restored_canary: i64 = restored_conn
        .query_row(
            "SELECT COUNT(*) FROM encryption_canary WHERE id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        restored_canary, 1,
        "restored encrypted database must retain its canary"
    );
    drop(restored_conn);
    assert_no_plaintext_bytes(
        &[restored_path.as_path()],
        &[old_body, new_body, post_rotation_body],
    );

    let backup_conn = rusqlite::Connection::open(&backup_path).unwrap();
    for table in [
        "entities",
        "entity_history",
        "entities_fts",
        "entity_history_fts",
    ] {
        let query = format!("SELECT body_json FROM {table}");
        let values: Vec<String> = backup_conn
            .prepare(&query)
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(
            values
                .iter()
                .all(|value| !value.contains(old_body) && !value.contains(new_body)),
            "backup {table} retained plaintext"
        );
    }
    drop(backup_conn);

    let backup_write = Command::new(BIN)
        .env("HOME", &home)
        .args([
            "write",
            "--db",
            backup_path.to_str().unwrap(),
            "--encryption-key",
            new_key.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "backup-open-check",
            "--body-json",
            r#"{"note":"backup-open-check-8c2e"}"#,
        ])
        .output()
        .expect("spawn backup open check");
    assert!(
        backup_write.status.success(),
        "backup could not be opened with the new key: {}",
        String::from_utf8_lossy(&backup_write.stderr)
    );
    assert_no_plaintext_bytes(
        &[db_path.as_path(), backup_path.as_path()],
        &[old_body, new_body, post_rotation_body],
    );

    for path in [&db_path, &backup_path] {
        for suffix in ["-wal", "-journal"] {
            let sidecar = PathBuf::from(format!("{}{}", path.display(), suffix));
            if sidecar.is_file() {
                let bytes = std::fs::read(&sidecar).unwrap();
                assert!(
                    !bytes
                        .windows(old_body.len())
                        .any(|window| window == old_body.as_bytes())
                        && !bytes
                            .windows(new_body.len())
                            .any(|window| window == new_body.as_bytes()),
                    "sidecar {} retained plaintext",
                    sidecar.display()
                );
            }
        }
    }

    let _ = std::fs::remove_dir_all(&home);
}
#[cfg(unix)]
#[test]
fn backup_rejects_broken_destination_symlink() {
    use std::os::unix::fs::symlink;

    let home = sandbox("backup-symlink");
    let db_path = home.join("vault.db");
    let destination = home.join("backup.db");
    let missing_target = home.join("missing-target.db");
    let write = Command::new(BIN)
        .env("HOME", &home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "backup-symlink",
            "--body-json",
            r#"{"note":"backup symlink probe"}"#,
        ])
        .output()
        .expect("spawn plaintext fixture write");
    assert!(
        write.status.success(),
        "fixture write failed: {}",
        String::from_utf8_lossy(&write.stderr)
    );
    symlink(&missing_target, &destination).unwrap();

    let backup = Command::new(BIN)
        .env("HOME", &home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
        .args([
            "backup",
            "--db",
            db_path.to_str().unwrap(),
            "--to",
            destination.to_str().unwrap(),
        ])
        .output()
        .expect("spawn symlink backup");
    assert!(
        !backup.status.success(),
        "backup must refuse a symlink destination"
    );
    assert!(
        destination.symlink_metadata().is_ok(),
        "destination symlink must remain"
    );
    assert!(
        !missing_target.exists(),
        "backup must not follow the broken symlink"
    );

    let _ = std::fs::remove_dir_all(&home);
}

#[cfg(unix)]
#[test]
fn backup_rejects_governance_source_symlink() {
    use std::os::unix::fs::symlink;

    let home = sandbox("backup-governance-source-symlink");
    let db_path = home.join("vault.db");
    let backup_path = home.join("backup.db");
    let foreign_overlay = home.join("foreign-overlay.db");
    let source_overlay = PathBuf::from(format!("{}.governance.db", db_path.display()));
    let write = Command::new(BIN)
        .env("HOME", &home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
        .args([
            "write",
            "--db",
            db_path.to_str().unwrap(),
            "--category",
            "facts",
            "--key",
            "backup-governance-source-symlink",
            "--body-json",
            r#"{"note":"backup governance source symlink probe"}"#,
        ])
        .output()
        .expect("spawn plaintext fixture write");
    assert!(
        write.status.success(),
        "fixture write failed: {}",
        String::from_utf8_lossy(&write.stderr)
    );
    {
        let conn = rusqlite::Connection::open(&foreign_overlay).unwrap();
        conn.execute_batch("CREATE TABLE overlay_marker (value TEXT NOT NULL);")
            .unwrap();
    }
    symlink(&foreign_overlay, &source_overlay).unwrap();

    let backup = Command::new(BIN)
        .env("HOME", &home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1")
        .args([
            "backup",
            "--db",
            db_path.to_str().unwrap(),
            "--to",
            backup_path.to_str().unwrap(),
        ])
        .output()
        .expect("spawn governance source symlink backup");
    assert!(
        !backup.status.success(),
        "backup must refuse a symlinked source governance sidecar"
    );
    assert!(
        source_overlay.symlink_metadata().is_ok(),
        "source governance symlink must remain"
    );
    assert!(
        !backup_path.exists(),
        "failed backup must not leave a primary destination"
    );
    assert!(
        !PathBuf::from(format!("{}.governance.db", backup_path.display())).exists(),
        "failed backup must not leave a sidecar destination"
    );

    let _ = std::fs::remove_dir_all(&home);
}
