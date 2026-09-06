//! #917: public MCP recall contradiction-flag surface.
//!
//! These tests drive the real binary so the assertions cover both the recall
//! handler and the MCP `structuredContent` projection. The fixture deliberately
//! uses operator writes (which are verified-grade records) and a keyword-only
//! recall so the tests do not depend on an embedding service.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

const BIN: &str = env!("CARGO_BIN_EXE_perseus-vault");

struct Fixture {
    root: PathBuf,
    db: PathBuf,
    home: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("perseus-vault-917-{name}-{}", uuid::Uuid::new_v4()));
        let home = root.join("home");
        let db = root.join("vault.db");
        std::fs::create_dir_all(&home).expect("fixture home");
        Self { root, db, home }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(BIN);
        command
            .env("HOME", &self.home)
            .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1");
        command
    }

    fn write(&self, category: &str, key: &str, content: &str) -> String {
        let body = json!({"content": content}).to_string();
        let output = self
            .command()
            .args([
                "write",
                "--db",
                self.db.to_str().expect("db path"),
                "--category",
                category,
                "--key",
                key,
                "--body-json",
                &body,
                "--importance",
                "0.95",
            ])
            .output()
            .expect("write entity");
        assert!(
            output.status.success(),
            "write failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).expect("write JSON");
        result["id"].as_str().expect("write id").to_string()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

struct Server {
    child: Child,
    stdin: std::process::ChildStdin,
    responses: Receiver<String>,
    next_id: u64,
}

impl Server {
    fn start(fixture: &Fixture) -> Self {
        let mut command = fixture.command();
        command
            .args([
                "serve",
                "--db",
                fixture.db.to_str().expect("db path"),
                "--offline",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().expect("start MCP server");
        let stdin = child.stdin.take().expect("server stdin");
        let stdout = child.stdout.take().expect("server stdout");
        let stderr = child.stderr.take().expect("server stderr");
        let (tx, responses) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) if tx.send(line.clone()).is_err() => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
        });
        std::thread::spawn(move || {
            // Keep stderr drained so a verbose child cannot block on its log pipe.
            let mut reader = BufReader::new(stderr);
            let mut discarded = String::new();
            while reader.read_line(&mut discarded).unwrap_or(0) > 0 {
                discarded.clear();
            }
        });
        let mut server = Self {
            child,
            stdin,
            responses,
            next_id: 1,
        };
        let initialized = server.request("initialize", json!({}));
        assert!(
            initialized["result"].is_object(),
            "initialize failed: {initialized}"
        );
        server.notify("notifications/initialized", json!({}));
        server
    }

    fn notify(&mut self, method: &str, params: Value) {
        let request = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{request}").expect("write notification");
        self.stdin.flush().expect("flush notification");
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        writeln!(self.stdin, "{request}").expect("write request");
        self.stdin.flush().expect("flush request");
        let line = self
            .responses
            .recv_timeout(Duration::from_secs(60))
            .expect("MCP response");
        serde_json::from_str(&line).expect("MCP response JSON")
    }

    fn call(&mut self, name: &str, arguments: Value) -> Value {
        let response = self.request("tools/call", json!({"name": name, "arguments": arguments}));
        assert!(
            response["result"].is_object(),
            "tool response missing result: {response}"
        );
        response["result"]["structuredContent"].clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn conflict_fixture(name: &str) -> (Fixture, String, String) {
    // Disjoint bodies: genuinely low trigram similarity, which is the
    // detector's contract for a divergent same-category pair (#917).
    let fixture = Fixture::new(name);
    let red = fixture.write("claim-fixture", "red", "alpha beta gamma delta epsilon");
    let blue = fixture.write("claim-fixture", "blue", "quantum xylophone jupiter nebula");
    (fixture, red, blue)
}

fn score_verified(server: &mut Server) {
    // score >= 0.7 sets verified=1 (db.rs score_entity) — the claim-card
    // high-confidence grade the flag surface gates on (#917 design).
    for key in ["red", "blue"] {
        server.call(
            "perseus_vault_score",
            json!({"category": "claim-fixture", "key": key, "score": 0.9}),
        );
    }
}
fn flags(value: &Value) -> &Vec<Value> {
    value["conflict_flags"]
        .as_array()
        .expect("opted-in recall must return conflict_flags")
}

#[test]
fn conflict_cluster_emits_both_sides_validity_and_hash_linked_refs() {
    let (fixture, red, blue) = conflict_fixture("cluster");
    let mut server = Server::start(&fixture);
    score_verified(&mut server);
    let result = server.call(
        "perseus_vault_recall",
        json!({
            "query": "alpha quantum",
            "category": "claim-fixture",
            "mode": "fts5",
            "limit": 10,
            "include_conflict_flags": true
        }),
    );
    let flags = flags(&result);
    assert!(
        flags
            .iter()
            .any(|flag| { flag["candidate_id"] == red && flag["claim_id"] == blue }),
        "red side missing: {result}"
    );
    assert!(
        flags
            .iter()
            .any(|flag| { flag["candidate_id"] == blue && flag["claim_id"] == red }),
        "blue side missing: {result}"
    );
    for flag in flags {
        assert_eq!(flag["kind"], "contradiction", "{flag}");
        assert_eq!(flag["confidence"], "high", "{flag}");
        assert_eq!(flag["disposition"], "flag", "{flag}");
        assert_eq!(flag["disclose_existence"], true, "{flag}");
        assert_eq!(flag["disclose_value"], false, "{flag}");
        assert!(flag["validity"]["candidate"]["valid_from_unix_ms"].is_number());
        assert!(flag["validity"]["claim"]["valid_from_unix_ms"].is_number());
        let refs = flag["evidence_refs"].as_array().expect("evidence refs");
        assert!(!refs.is_empty(), "{flag}");
        for evidence in refs {
            assert!(evidence["entity_id"].is_string(), "{evidence}");
            assert!(evidence["card_digest"].is_string(), "{evidence}");
            assert!(
                evidence.get("content").is_none(),
                "raw claim leaked: {evidence}"
            );
        }
    }
    assert_eq!(result["abstain_hint"], true, "{result}");
}

#[test]
fn clean_recall_has_empty_conflict_flags_and_no_abstain_hint() {
    let fixture = Fixture::new("clean");
    fixture.write("clean-fixture", "clean", "clean");
    let mut server = Server::start(&fixture);
    let result = server.call(
        "perseus_vault_recall",
        json!({
            "query": "clean",
            "category": "clean-fixture",
            "mode": "fts5",
            "include_conflict_flags": true
        }),
    );
    assert!(flags(&result).is_empty(), "{result}");
    assert_eq!(result["abstain_hint"], false, "{result}");
}

#[test]
fn suppressed_claim_discloses_only_existence_and_never_its_value() {
    let fixture = Fixture::new("suppressed");
    let visible = fixture.write("suppressed-fixture", "visible", "visible");
    let suppressed = fixture.write("suppressed-fixture", "hidden", "forbidden-secret");
    let mut server = Server::start(&fixture);
    let rejected = server.call(
        "perseus_vault_reject_value",
        json!({
            "workspace_hash": "",
            "subject": "hidden",
            "predicate": "suppressed-fixture",
            "value": "{\"content\":\"forbidden-secret\"}",
            "reason": "#917 test"
        }),
    );
    assert_eq!(rejected["rejected"], true, "{rejected}");

    let result = server.call(
        "perseus_vault_recall",
        json!({
            "query": "visible forbidden-secret",
            "category": "suppressed-fixture",
            "mode": "fts5",
            "include_conflict_flags": true
        }),
    );
    let flags = flags(&result);
    let flag = flags
        .iter()
        .find(|flag| flag["claim_id"] == suppressed || flag["candidate_id"] == suppressed)
        .expect("suppressed claim existence flag");
    assert_eq!(flag["disclose_existence"], true, "{flag}");
    assert_eq!(flag["disclose_value"], false, "{flag}");
    assert_eq!(flag["claim_id"], suppressed, "{flag}");
    assert!(
        result.to_string().contains(&visible),
        "visible side should remain inspectable"
    );
    assert!(
        !result.to_string().contains("forbidden-secret"),
        "suppressed value leaked: {result}"
    );
}

fn recall_mutation_summary(path: &Path) -> (Vec<(String, String, i64, String)>, i64, i64) {
    let conn = rusqlite::Connection::open(path).expect("open recall summary database");
    let mut statement = conn
        .prepare(
            "SELECT category, key, retrieval_count, layer
             FROM entities ORDER BY category, key",
        )
        .expect("prepare recall summary");
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .expect("query recall summary")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect recall summary");
    let served_events: i64 = conn
        .query_row("SELECT COUNT(*) FROM served_events", [], |row| row.get(0))
        .expect("count served events");
    let arm_audits: i64 = conn
        .query_row("SELECT COUNT(*) FROM recall_arm_audits", [], |row| {
            row.get(0)
        })
        .expect("count recall arm audits");
    (rows, served_events, arm_audits)
}

#[test]
fn flagged_recall_adds_no_mutation_beyond_the_existing_recall_contract() {
    // Ordinary FTS recall intentionally reinforces returned rows and records
    // retrieval telemetry. Compare independent, otherwise identical fixtures
    // so this test proves the flag projection adds no writes beyond that
    // pre-existing contract rather than incorrectly requiring byte identity.
    let (plain_fixture, _plain_red, _plain_blue) = conflict_fixture("readonly-plain");
    let (flagged_fixture, _flagged_red, _flagged_blue) = conflict_fixture("readonly-flagged");

    let mut plain_server = Server::start(&plain_fixture);
    score_verified(&mut plain_server);
    plain_server.call(
        "perseus_vault_recall",
        json!({
            "query": "alpha quantum",
            "category": "claim-fixture",
            "mode": "fts5"
        }),
    );
    drop(plain_server);

    let mut flagged_server = Server::start(&flagged_fixture);
    score_verified(&mut flagged_server);
    let result = flagged_server.call(
        "perseus_vault_recall",
        json!({
            "query": "alpha quantum",
            "category": "claim-fixture",
            "mode": "fts5",
            "include_conflict_flags": true
        }),
    );
    assert!(
        !flags(&result).is_empty(),
        "fixture must exercise flags: {result}"
    );
    drop(flagged_server);

    assert_eq!(
        recall_mutation_summary(&plain_fixture.db),
        recall_mutation_summary(&flagged_fixture.db),
        "flag emission must not add mutation beyond ordinary FTS recall"
    );
}

#[test]
fn abstain_hint_is_false_without_a_high_confidence_direct_contradiction() {
    let fixture = Fixture::new("threshold");
    fixture.write("threshold-fixture", "one", "one");
    let mut server = Server::start(&fixture);
    let result = server.call(
        "perseus_vault_recall",
        json!({
            "query": "one",
            "category": "threshold-fixture",
            "mode": "fts5",
            "include_conflict_flags": true
        }),
    );
    assert!(flags(&result).is_empty(), "{result}");
    assert_eq!(result["abstain_hint"], false, "{result}");
}

#[test]
fn conflict_flags_are_default_off_and_markdown_is_independently_opt_in() {
    let (fixture, _red, _blue) = conflict_fixture("toggle");
    let mut server = Server::start(&fixture);
    let default_result = server.call(
        "perseus_vault_recall",
        json!({
            "query": "alpha quantum",
            "category": "claim-fixture",
            "mode": "fts5"
        }),
    );
    assert!(
        default_result.get("conflict_flags").is_none(),
        "{default_result}"
    );
    assert!(
        default_result.get("abstain_hint").is_none(),
        "{default_result}"
    );
    assert!(
        default_result.get("conflict_flags_markdown").is_none(),
        "{default_result}"
    );

    let markdown_result = server.call(
        "perseus_vault_recall",
        json!({
            "query": "alpha quantum",
            "category": "claim-fixture",
            "mode": "fts5",
            "include_conflict_flags_markdown": true
        }),
    );
    assert!(
        markdown_result["conflict_flags_markdown"]
            .as_str()
            .is_some_and(|text| text.contains("conflict")),
        "{markdown_result}"
    );
    assert!(
        markdown_result.get("conflict_flags").is_none(),
        "markdown-only mode must not implicitly add JSON flags: {markdown_result}"
    );
}

#[test]
fn mcp_schema_advertises_opt_in_conflict_flag_arguments() {
    let fixture = Fixture::new("schema");
    let mut server = Server::start(&fixture);
    let response = server.request("tools/list", json!({}));
    let tools = response["result"]["tools"].as_array().expect("tools list");
    let recall = tools
        .iter()
        .find(|tool| tool["name"] == "perseus_vault_recall")
        .expect("recall schema");
    let properties = &recall["inputSchema"]["properties"];
    assert_eq!(properties["include_conflict_flags"]["type"], "boolean");
    assert_eq!(properties["include_conflict_flags"]["default"], false);
    assert_eq!(
        properties["include_conflict_flags_markdown"]["type"],
        "boolean"
    );
    assert_eq!(
        properties["include_conflict_flags_markdown"]["default"],
        false
    );
}

#[allow(dead_code)]
fn _assert_path_is_file(path: &Path) {
    assert!(path.is_file(), "expected file at {}", path.display());
}
