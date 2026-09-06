//! #968: real-process crash recovery for durable op runs.
//!
//! Unit tests cover the mark-on-open recovery path in-process; this test
//! exercises it against an actually-killed process: spawn the real binary
//! running `decay` on a seeded store, SIGKILL it mid-run (the durable
//! `running` state is observed first), reopen — the run must be
//! `interrupted` — and an explicit `op-runs retry` must fork a re-queued
//! child run (the resume path from #871).
//!
//! Polling uses a raw SQLite connection on purpose: `Database::open` runs
//! `op_runs::recover`, so a second CLI process would mark the live run
//! `interrupted` mid-flight. Raw reads observe the durable state without
//! side effects.
//!
//! Item note: the CLI `decay`/`maintain` passes are item-less by design
//! (retry on them is honestly `nothing_to_retry`). The retry contract is
//! item-based, so after the kill we inject one in-flight item via SQL —
//! exactly the state a fan-out op (consolidate/export) would leave — and
//! assert recovery marks it `interrupted` and retry re-queues it into the
//! child run.

use std::process::Command;
use std::time::{Duration, Instant};

use rusqlite::OptionalExtension;

const BIN: &str = env!("CARGO_BIN_EXE_perseus-vault");

fn scratch_db(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("{name}-{}.db", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

fn scratch_home(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("{name}-home-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create test HOME");
    path
}

fn test_command(home: &std::path::Path) -> Command {
    let mut command = Command::new(BIN);
    // `init` now creates an encrypted store by default. This test deliberately
    // seeds through raw SQL to exercise op-run crash recovery, not encryption;
    // use an isolated HOME with no key and explicitly permit the plaintext
    // fixture so each child reaches op_run_start before the 300k-row workload.
    command
        .env("HOME", home)
        .env("PERSEUS_VAULT_ALLOW_PLAINTEXT", "1");
    command
}

fn op_runs(db: &std::path::Path, home: &std::path::Path, args: &[&str]) -> (bool, String) {
    let out = test_command(home)
        .arg("op-runs")
        .arg("--db")
        .arg(db.to_str().unwrap())
        .args(args)
        .output()
        .expect("spawn perseus-vault");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
    )
}

#[test]
fn op_runs_kill9_mid_decay_marks_interrupted_and_retry_recovers() {
    let db = scratch_db("opruns-kill9");
    let home = scratch_home("opruns-kill9");

    // 1) Initialize the vault so the DB exists with the current schema.
    let init = test_command(&home)
        .arg("init")
        .arg("--db")
        .arg(db.to_str().unwrap())
        .output()
        .expect("init");
    assert!(init.status.success(), "init failed");

    // The stress rows are intentionally plaintext fixture data. Remove only the
    // empty init markers and its isolated key; no production database is touched.
    {
        let conn = rusqlite::Connection::open(&db).expect("fixture connection");
        conn.execute_batch("DELETE FROM encryption_canary; DELETE FROM encryption_profile;")
            .expect("clear empty encryption fixture markers");
    }
    std::fs::remove_file(home.join(".perseus-vault/secret.key")).expect("remove fixture key");

    // 2) Seed a store large enough that the decay pass runs for seconds.
    //    Insert shape mirrors src/db.rs `seed_bulk_entities` — keep in sync
    //    if the entities schema changes. Rows have last_accessed=0 so every
    //    row is decay-due; decay_tick processes all non-archived rows.
    {
        let conn = rusqlite::Connection::open(&db).expect("seed connection");
        conn.execute_batch("PRAGMA busy_timeout=5000;").unwrap();
        let filler = "x".repeat(360);
        let tx = conn.unchecked_transaction().unwrap();
        for i in 0..300_000u32 {
            tx.execute(
                "INSERT INTO entities (id, category, key, body_json, status, type, tags, \
                 decay_score, retrieval_count, layer, topic_path, archived, archive_reason, \
                 links, verified, source, created_at_unix_ms, last_accessed_unix_ms) \
                 VALUES (?1, 'bulk', ?2, ?3, 'active', 'insight', '[\"bench\"]', 0.5, 0, \
                 'working', '', 0, '', '[]', 0, 'agent', 0, 0)",
                rusqlite::params![
                    format!("k9-{i:06}"),
                    format!("k9-key-{i:06}"),
                    format!("{{\"content\":\"kill9 row {i}\",\"filler\":\"{filler}\"}}"),
                ],
            )
            .unwrap();
        }
        tx.commit().unwrap();
    }

    // 3) Spawn `decay` — a durable op run (begin -> start -> ... -> complete).
    let mut child = test_command(&home)
        .arg("decay")
        .arg("--db")
        .arg(db.to_str().unwrap())
        .spawn()
        .expect("spawn decay");

    // 4) Wait for the durable run to enter `running` (raw read — no
    //    recovery side effects; see module docs).
    let run_id: String = {
        let conn = rusqlite::Connection::open(&db).expect("poll connection");
        conn.execute_batch("PRAGMA busy_timeout=5000;").unwrap();
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let row = conn
                .query_row(
                    "SELECT id, state FROM op_runs WHERE op_type='decay' \
                     ORDER BY created_at_unix_ms DESC LIMIT 1",
                    [],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
                )
                .optional()
                .expect("poll op_runs");
            if let Some((id, state)) = row {
                if state == "running" {
                    break id;
                }
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("decay never entered the durable running state");
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    };

    // 5) SIGKILL mid-run (Child::kill = SIGKILL on unix, TerminateProcess on
    //    windows). The pass must NOT have completed.
    child.kill().expect("kill decay");
    child.wait().expect("wait decay");

    // 5.5) Inject one in-flight item, as a fan-out op (consolidate/export)
    //      would have left at kill time. Recovery below must mark it
    //      `interrupted` alongside the run.
    {
        let conn = rusqlite::Connection::open(&db).expect("item connection");
        conn.execute(
            "INSERT INTO op_run_items (id, run_id, item_ref, state, item_digest, receipt_ref, \
             error_class, error_detail, retry_count, created_at_unix_ms, updated_at_unix_ms, \
             finished_at_unix_ms) \
             VALUES (?1, ?2, ?3, 'running', 'digest-k9', '', '', '', 0, 0, 0, NULL)",
            rusqlite::params![
                format!("opi-k9-{}", std::process::id()),
                run_id,
                "fanout-unit-1",
            ],
        )
        .expect("inject in-flight item");
    }

    // 6) Reopen via the CLI: restart recovery marks the dead run (and its
    //    in-flight item) interrupted.
    let (ok, list) = op_runs(&db, &home, &[]);
    assert!(ok, "list failed");
    assert!(
        list.contains("interrupted"),
        "run must be interrupted after restart: {list}"
    );

    // 7) Explicit retry forks a re-queued child run and re-queues the
    //    interrupted item (idempotent resume). Verified with raw reads:
    //    any further CLI open would run restart recovery, which by design
    //    marks a still-`queued` child `interrupted` (mark-only; a worker
    //    resumes it) and would mask the retry outcome.
    let out = test_command(&home)
        .arg("op-runs")
        .arg("--db")
        .arg(db.to_str().unwrap())
        .args(["--action", "retry", "--run-id", &run_id])
        .output()
        .expect("retry");
    assert!(
        out.status.success(),
        "retry failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The child run exists, links to the interrupted parent, and is queued.
    let child_id: String = {
        let conn = rusqlite::Connection::open(&db).expect("child connection");
        conn.query_row(
            "SELECT id FROM op_runs WHERE parent_run_id = ?1",
            rusqlite::params![run_id],
            |r| r.get(0),
        )
        .expect("child run exists")
    };
    {
        let conn = rusqlite::Connection::open(&db).expect("child state connection");
        let child_state: String = conn
            .query_row(
                "SELECT state FROM op_runs WHERE id = ?1",
                rusqlite::params![child_id],
                |r| r.get(0),
            )
            .expect("child state");
        assert_eq!(
            child_state, "queued",
            "child run must be queued for a worker"
        );
    }
    // The child run carries the re-queued item.
    {
        let conn = rusqlite::Connection::open(&db).expect("item check connection");
        let item_state: String = conn
            .query_row(
                "SELECT state FROM op_run_items WHERE run_id = ?1 AND item_ref = 'fanout-unit-1'",
                rusqlite::params![child_id],
                |r| r.get(0),
            )
            .expect("child item re-queued");
        assert_eq!(
            item_state, "queued",
            "item must be re-queued in the child run"
        );
    }

    let _ = std::fs::remove_file(&db);
    let _ = std::fs::remove_dir_all(&home);
}
