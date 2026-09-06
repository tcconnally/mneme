use rusqlite::{params, Connection, OptionalExtension};
use serde_json::json;

use crate::db::now_ms;
use crate::encryption::EncryptionManager;
use crate::models::{MigrationReport, Stats};

/// SQL to create the v0.2.0 schema from scratch.
pub const DDL_V0_2_0: &str = "
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL DEFAULT 'general',
    key TEXT NOT NULL,
    body_json TEXT NOT NULL DEFAULT '{}',
    status TEXT DEFAULT 'active',
    type TEXT DEFAULT 'insight',
    tags TEXT DEFAULT '[]',
    decay_score REAL DEFAULT 1.0,
    retrieval_count INTEGER DEFAULT 0,
    layer TEXT DEFAULT 'working',
    topic_path TEXT DEFAULT '',
    archived INTEGER DEFAULT 0,
    archive_reason TEXT DEFAULT '',
    links TEXT DEFAULT '[]',
    verified INTEGER DEFAULT 0,
    source TEXT DEFAULT 'agent',
    created_at_unix_ms INTEGER NOT NULL,
    last_accessed_unix_ms INTEGER NOT NULL,
    embedding BLOB,
    -- Sign-bit signature of `embedding` (v2.13.0, dim/8 bytes): bit i set iff
    -- embedding[i] > 0. dense_search Hamming-prefilters on this instead of
    -- reading every full embedding blob once the vault is large enough.
    -- Written by store_embedding; backfilled by the v6 migration.
    emb_sig BLOB,
    always_on INTEGER DEFAULT 0,
    certainty REAL DEFAULT 0.5,
    -- Persistent importance floor (v2.13.0). Set by perseus_vault_score; decay_tick and
    -- cohere floor decay_score at this value, so an explicit score survives the
    -- recency-based recompute instead of being erased by the next tick
    -- (fidelity > recency). 0.0 = unset, no effect.
    importance REAL DEFAULT 0.0,
    workspace_hash TEXT DEFAULT '',
    agent_id TEXT DEFAULT '',
    visibility TEXT DEFAULT 'workspace',
    -- Bi-temporal facts (v2.4.0). Two time axes plus a supersession link, so a
    -- fact can be retired without deleting history. All NULL/'' here means
    -- \"valid since creation, currently true, never superseded\" — the behavior
    -- before bi-temporal support, so existing rows need no interpretation change.
    valid_from_unix_ms INTEGER,      -- when the fact became true in the world (NULL = since creation)
    valid_to_unix_ms INTEGER,        -- when it stopped being true (NULL = still true)
    recorded_at_unix_ms INTEGER,     -- transaction time: when Perseus Vault first knew it (backfilled = created_at)
    invalidated_at_unix_ms INTEGER,  -- transaction time: when Perseus Vault retired it (NULL = live)
    supersedes TEXT DEFAULT '',      -- id of the entity this one replaced
    superseded_by TEXT DEFAULT '',   -- id of the entity that replaced this one

    -- Efficacy tracking (v2.10.0 — PMB-inspired follow-rate scoring). Tracks
    -- whether a lesson/convention/insight actually gets FOLLOWED by the agent,
    -- not just recalled. follow_rate feeds into decay_tick as a composite
    -- weight so rules that get ignored decay out of recall, and rules that
    -- earn their place resist decay even without recency.
    follow_count INTEGER DEFAULT 0,      -- times confirmed/detected as followed
    miss_count INTEGER DEFAULT 0,        -- times confirmed/detected as NOT followed
    follow_rate REAL DEFAULT 0.0,        -- follow_count / (follow_count + miss_count), 0.0 if no attempts
    efficacy_status TEXT DEFAULT 'unverified',  -- 'unverified' | 'useful' | 'dead'

    -- Epistemic trust axis (#880 — orthogonal to the lifecycle `status` axis).
    -- status says where the record sits in its life (active/proposed/
    -- quarantined/superseded/…); epistemic_state says how much it may be
    -- TRUSTED as fact: 'candidate' (useful but unverified), 'verified'
    -- (authoritative admission or operator promotion), 'corroborated'
    -- (multiple independent evidence refs), 'rejected' (reviewed and refused),
    -- 'defensively_recalled' (served despite low trust, explicitly framed as
    -- untrusted). Default 'candidate': a fresh write is useful-but-unverified
    -- until admission or promotion proves it.
    epistemic_state TEXT DEFAULT 'candidate',

    -- Usefulness tracking (#487 — Belief-Memory-inspired derived_from
    -- reinforcement). Unlike retrieval_count (how often a memory was merely
    -- recalled), usefulness_count only increments when a later remember()
    -- explicitly cites this entity via derived_from — i.e. the memory
    -- demonstrably informed a subsequent write. Feeds decay_tick (cited
    -- memories decay slower) and the hybrid-recall rank boost.
    usefulness_count INTEGER DEFAULT 0,  -- times cited as a derived_from source
    last_useful_unix_ms INTEGER DEFAULT 0, -- when it was last cited (0 = never)

    -- Retention expiry (#868): when this fact stops being serveable.
    -- Written by the remember path from the body `expires_at` convention
    -- (unix ms, numeric string, or ISO 8601 UTC); recall excludes rows past
    -- it, and the expire sweep transitions them to status='expired'. NULL =
    -- never expires (the default for every pre-existing row).
    expires_at_unix_ms INTEGER,

    -- Learned-anticipation tuning marker (#875): set to the apply time of a
    -- governed preload_review approve on this entity. A tuning write rewrites
    -- the body (and bumps last_accessed), so usage signal gathered before it
    -- must not be read as fresh usage: add_trigger proposals skip entities
    -- whose tuning is newer than their latest preload serve, until the
    -- entity is re-observed. 0 = never tuned.
    preload_tuned_unix_ms INTEGER DEFAULT 0
);

-- Identity index: (category, key, workspace_hash) — #339. Created in
-- initialize_schema's gated block, NOT here: on a legacy DB this ungated DDL
-- runs before the ALTER that adds workspace_hash, so an index referencing the
-- column here would fail the whole batch.

-- Recall ranking index: lets the browse path (WHERE archived=0 [+ residual
-- filters] ORDER BY retrieval_count DESC, last_accessed_unix_ms DESC, id ASC
-- LIMIT k) seek the archived=0 partition and read rows already in rank order,
-- avoiding a full table scan + temp-b-tree sort. EXPLAIN-verified: ~224x on
-- global browse, ~66x on workspace-scoped browse at 30k rows. (#209)
-- The trailing `id ASC` covers recall's #254 determinism tie-break: without it,
-- a large tie-group on (retrieval_count, last_accessed) — e.g. a cold or
-- bulk-imported store with uniform last_accessed — forced SQLite to sort the
-- whole group by id to satisfy LIMIT k (O(tie-group); ~30ms browse @1M). With it
-- the index satisfies the FULL ordering, so browse stays a k-row range scan.
CREATE INDEX IF NOT EXISTS idx_entities_recall ON entities(archived, retrieval_count DESC, last_accessed_unix_ms DESC, id ASC);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(body_json, content_rowid='rowid');

CREATE TABLE IF NOT EXISTS journal (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL DEFAULT 'decision',
    evaluated_json TEXT DEFAULT '{}',
    acted_json TEXT DEFAULT '{}',
    forward_json TEXT DEFAULT '{}',
    category TEXT DEFAULT '',
    key TEXT DEFAULT '',
    entity_id TEXT DEFAULT '',
    agent_id TEXT DEFAULT '',
    audit_hash TEXT DEFAULT '',
    -- #417: workspace of the referenced entity, stamped at write time so purge
    -- can scope journal redaction per-workspace. '' = system event or legacy row.
    workspace_hash TEXT NOT NULL DEFAULT '',
    -- v15 (2026-07-05): SHA-256 commitment over the payload, covered by the audit
    -- chain so content tampering is detectable while the payload can still be
    -- redacted (the commitment survives). See docs/audit-chain-keyed-mac-design.md.
    payload_commitment TEXT DEFAULT '',
    -- v23 (Chancery cross-ref, #6): writ ID from Chancery's `_meta` envelope,
    -- so vault journal entries can be joined to Chancery writ records.
    chancery_writ_id TEXT DEFAULT '',
    created_at_unix_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_journal_created ON journal(created_at_unix_ms);
CREATE INDEX IF NOT EXISTS idx_journal_entity ON journal(entity_id);
-- NOTE: the (category, key, workspace_hash) index is created in apply_migrations,
-- not here -- on a legacy DB this batch runs BEFORE the workspace_hash column is
-- added, so referencing it here would fail with a no-such-column error (#417).

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL DEFAULT '{}',
    expires_at_unix_ms INTEGER,
    created_at_unix_ms INTEGER NOT NULL
);

-- Encryption canary (fail-fast wrong-key detection). A single row (id=1) holding
-- a known marker encrypted under the configured key. On startup set_encryption
-- decrypts it and aborts loudly if it fails, so a wrong/rotated key is caught
-- before any read silently returns AuthFailed. Deliberately NOT in `entities` or
-- `state`: it must stay invisible to recall/FTS/stats and to caller-facing state
-- tools. Ungated IF NOT EXISTS so it back-fills onto older databases too.
CREATE TABLE IF NOT EXISTS encryption_canary (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    ciphertext TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL
);

-- Protected searchable-index metadata. The table is present on every store.
-- `migration-pending` is a non-activating crash-recovery marker written while
-- canonical bodies/FTS are being rewritten; only the singleton row with the
-- keyed blind-token mode advertises completed protected search.
CREATE TABLE IF NOT EXISTS encryption_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    search_mode TEXT NOT NULL,
    updated_at_unix_ms INTEGER NOT NULL
);

-- v15 (2026-07-05): records the audit chain keying so set_encryption can skip
-- the O(journal) rekey when the chain is already keyed under the current key.
-- key_canary = HMAC-SHA256(audit_key, fixed label); a match means already-keyed
-- under this key. See docs/audit-chain-keyed-mac-design.md section 3.4.
CREATE TABLE IF NOT EXISTS audit_chain_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    scheme TEXT NOT NULL,
    key_canary TEXT NOT NULL,
    updated_at_unix_ms INTEGER NOT NULL
);

-- Superseded fact versions (v2.4.0 — bi-temporal facts). When a remember()
-- overwrites an existing (category,key,workspace_hash) with new content, the prior
-- row is snapshotted here with invalidated_at set, so live reads stay one-row-per-key
-- (entities + its UNIQUE(category,key,workspace_hash) are untouched) while history is kept for
-- as-of / time-travel queries. A version was live during
-- [recorded_at_unix_ms, invalidated_at_unix_ms). superseded_by points at the
-- live entity id that replaced it. body_json carries the same encryption as
-- entities (ciphertext if a key is configured).
CREATE TABLE IF NOT EXISTS entity_history (
    history_id TEXT PRIMARY KEY,
    id TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    key TEXT NOT NULL,
    body_json TEXT NOT NULL DEFAULT '{}',
    status TEXT DEFAULT 'active',
    type TEXT DEFAULT 'insight',
    tags TEXT DEFAULT '[]',
    decay_score REAL DEFAULT 1.0,
    retrieval_count INTEGER DEFAULT 0,
    layer TEXT DEFAULT 'working',
    topic_path TEXT DEFAULT '',
    archived INTEGER DEFAULT 0,
    archive_reason TEXT DEFAULT '',
    links TEXT DEFAULT '[]',
    verified INTEGER DEFAULT 0,
    source TEXT DEFAULT 'agent',
    always_on INTEGER DEFAULT 0,
    certainty REAL DEFAULT 0.5,
    workspace_hash TEXT DEFAULT '',
    agent_id TEXT DEFAULT '',
    visibility TEXT DEFAULT 'workspace',
    valid_from_unix_ms INTEGER,
    valid_to_unix_ms INTEGER,
    recorded_at_unix_ms INTEGER,
    invalidated_at_unix_ms INTEGER,
    supersedes TEXT DEFAULT '',
    superseded_by TEXT DEFAULT '',
    created_at_unix_ms INTEGER NOT NULL,
    last_accessed_unix_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entity_history_id ON entity_history(id);
CREATE INDEX IF NOT EXISTS idx_entity_history_catkey ON entity_history(category, key, invalidated_at_unix_ms);

-- #682 Temporal RAG: a standalone FTS5 index over superseded/retired body terms.
-- entities_fts only covers LIVE rows, so a point-in-time semantic query could
-- never surface a fact whose query-matching version had since been superseded
-- (its text now lives only in entity_history) — the documented v1 limitation.
-- This index is queried ONLY when a temporal filter is active, purely to
-- discover (category,key) candidates the live index missed; the authoritative
-- point-in-time reconstruction still runs through bitemporal_at/as_of. Rowids
-- mirror entity_history.rowid, maintained at the single history-append site and
-- cleared at the two history-delete sites (purge/forget) to avoid rowid reuse.
CREATE VIRTUAL TABLE IF NOT EXISTS entity_history_fts USING fts5(body_json);

-- #683 Keystones: mandatory policy rules, fetched deterministically at session
-- start (perseus_vault_keystone_get) and obeyed over conflicting instructions. Merged
-- across scope (tenant < fleet < agent) with weight-based conflict resolution.
-- UNIQUE(scope, scope_id, content, workspace_hash) makes re-setting the same
-- rule an in-place update rather than a duplicate. Mutations are appended to
-- the cryptographic audit chain (like entity ops).
CREATE TABLE IF NOT EXISTS keystones (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'tenant',
    scope_id TEXT NOT NULL DEFAULT '',
    weight REAL NOT NULL DEFAULT 1.0,
    trust_tier_required INTEGER NOT NULL DEFAULT 2,
    workspace_hash TEXT NOT NULL DEFAULT '',
    author_agent_id TEXT NOT NULL DEFAULT '',
    created_at_unix_ms INTEGER NOT NULL,
    updated_at_unix_ms INTEGER NOT NULL,
    UNIQUE(scope, scope_id, content, workspace_hash)
);
CREATE INDEX IF NOT EXISTS idx_keystones_scope
    ON keystones(workspace_hash, scope, scope_id, weight);

-- #684 Multi-agent scoping: the agent registry. entities/journal already carry
-- agent_id (v1.2.0); this adds identity metadata + a trust tier (0-3) that gates
-- sensitive ops and drives visibility enforcement on reads. tier 0 = own only,
-- 1 = fleet, 2 = all + author keystones, 3 = admin. Unregistered agent_ids
-- resolve to tier 0. An empty agent_id (no session identity) is treated as
-- unscoped/admin so single-agent deployments are unaffected.
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    trust_tier INTEGER NOT NULL DEFAULT 0,
    fleet_id TEXT NOT NULL DEFAULT '',
    created_at_unix_ms INTEGER NOT NULL,
    updated_at_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agents_fleet ON agents(fleet_id);

-- #768 Authorized Action Receipts control plane. Manifests are versioned rather
-- than mutated, and actions are append-only state records whose transitions are
-- additionally written to the keyed journal audit chain.
CREATE TABLE IF NOT EXISTS authority_manifests (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    workspace_hash TEXT NOT NULL,
    version INTEGER NOT NULL,
    allowed_capabilities TEXT NOT NULL DEFAULT '[]',
    approval_required_capabilities TEXT NOT NULL DEFAULT '[]',
    scope_anchors TEXT NOT NULL DEFAULT '[]',
    approver_principals TEXT NOT NULL DEFAULT '[]',
    allowed_inbound_principals TEXT NOT NULL DEFAULT '[]',
    permitted_external_ref_prefixes TEXT NOT NULL DEFAULT '[]',
    max_parallel_actions INTEGER NOT NULL DEFAULT 1,
    mode TEXT NOT NULL DEFAULT 'shadow',
    expires_at_unix_ms INTEGER,
    revoked_at_unix_ms INTEGER,
    created_at_unix_ms INTEGER NOT NULL,
    UNIQUE(agent_id, workspace_hash, version)
);
CREATE INDEX IF NOT EXISTS idx_authority_active
 ON authority_manifests(agent_id, workspace_hash, revoked_at_unix_ms, version DESC);
-- #997: durable principal-revocation ledger. A revocation row subtracts the
-- principal from every ACTIVE manifest grant set (credential-relative cutoff:
-- a revocation stamped at >= the manifest's mint time bites for the
-- credential's full lifetime; reinstatement is an explicit, durable act).
CREATE TABLE IF NOT EXISTS revocations (
    id TEXT PRIMARY KEY,
    principal TEXT NOT NULL,
    workspace_hash TEXT NOT NULL DEFAULT '',
    at_unix_ms INTEGER NOT NULL,
    reinstated_at_unix_ms INTEGER,
    reason TEXT NOT NULL DEFAULT '',
    recorded_at_unix_ms INTEGER NOT NULL,
    UNIQUE(principal, workspace_hash, at_unix_ms)
);
CREATE INDEX IF NOT EXISTS idx_revocations_active
 ON revocations(workspace_hash, principal, reinstated_at_unix_ms);
CREATE TABLE IF NOT EXISTS authorized_actions (
    id TEXT PRIMARY KEY,
    manifest_id TEXT NOT NULL REFERENCES authority_manifests(id),
    manifest_version INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    workspace_hash TEXT NOT NULL,
    scope_anchor TEXT NOT NULL,
    external_ref TEXT NOT NULL DEFAULT '',
    capability TEXT NOT NULL,
    action_key TEXT NOT NULL,
    intent_hash TEXT NOT NULL,
    outcome_hash TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    approval_required INTEGER NOT NULL DEFAULT 0,
    approval_ref TEXT NOT NULL DEFAULT '',
    created_at_unix_ms INTEGER NOT NULL,
    updated_at_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_authorized_actions_scope
 ON authorized_actions(workspace_hash, action_key, status);
CREATE TABLE IF NOT EXISTS authorized_action_leases (
    id TEXT PRIMARY KEY,
    action_id TEXT NOT NULL REFERENCES authorized_actions(id),
    workspace_hash TEXT NOT NULL,
    action_key TEXT NOT NULL,
    holder_id TEXT NOT NULL,
    expires_at_unix_ms INTEGER NOT NULL,
    released_at_unix_ms INTEGER,
    created_at_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_authorized_action_leases_active
 ON authorized_action_leases(workspace_hash, action_key, released_at_unix_ms, expires_at_unix_ms);

-- #811 Immutable artifacts: content-addressed bytes live once in `artifacts`,
-- while scope/provenance/representation metadata lives in `artifact_bindings` so
-- the same bytes can be visible in multiple workspaces without leaking access.
-- Digest semantics (#835): sha256 is byte identity only — it proves the bytes
-- are the bytes; it says nothing about logical content, validity, authority,
-- or freshness, which come from binding/entity state, never the digest.
CREATE TABLE IF NOT EXISTS artifacts (
    sha256 TEXT PRIMARY KEY,
    content_b64 TEXT NOT NULL,
    byte_length INTEGER NOT NULL,
    created_at_unix_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_bindings (
    binding_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
    mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
    workspace_hash TEXT NOT NULL DEFAULT '',
    agent_id TEXT NOT NULL DEFAULT '',
    visibility TEXT NOT NULL DEFAULT 'workspace',
    origin_json TEXT NOT NULL DEFAULT '{}',
    external_refs_json TEXT NOT NULL DEFAULT '[]',
    retention_policy TEXT NOT NULL DEFAULT '',
    representation_kind TEXT NOT NULL DEFAULT 'original',
    derived_from_sha256 TEXT NOT NULL DEFAULT '',
    derivation_kind TEXT NOT NULL DEFAULT '',
    derivation_version TEXT NOT NULL DEFAULT '',
    created_at_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifact_bindings_scope
 ON artifact_bindings(sha256, workspace_hash, visibility, created_at_unix_ms DESC);
CREATE INDEX IF NOT EXISTS idx_artifact_bindings_derived_from
 ON artifact_bindings(derived_from_sha256);

-- #885 Optional quantized embedding storage: the store-wide `entities.embedding`
-- format record (single row, written by the reindex path and mirrored at open
-- for fresh stores) and the pre-quantization float32 snapshot used by the
-- documented rollback path (perseus_vault_embed restore_quantized_backup).
-- The snapshot is a migration artifact owned by the operator: it exists only
-- between `quantize` and the operator-confirmed drop, and nothing else reads
-- or writes it.
CREATE TABLE IF NOT EXISTS embedding_format (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format TEXT NOT NULL,
    updated_at_unix_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS entities_embedding_snapshot (
    id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    created_at_unix_ms INTEGER NOT NULL
);

-- #990 Deletion-residue accounting: declared basis for tier-3 projections
-- (embeddings first). Each row declares what a derived surface was built
-- from, so purge can classify residue and the independent sweep can verify
-- that the undeclared-residual cell stays empty. `source_entity_id = ''`
-- marks a bulk projection (e.g. the quantization snapshot), which is exempt
-- from the orphan sweep. See docs/specs/deletion-residue-accounting.md.
CREATE TABLE IF NOT EXISTS projection_basis (
    projection_kind            TEXT NOT NULL,
    projection_id              TEXT NOT NULL,
    source_entity_id           TEXT NOT NULL DEFAULT '',
    source_digest              TEXT NOT NULL DEFAULT '',
    source_recorded_at_unix_ms INTEGER NOT NULL DEFAULT 0,
    built_at_unix_ms           INTEGER NOT NULL,
    content_class              TEXT NOT NULL DEFAULT 'derived_content',
    transform                  TEXT NOT NULL DEFAULT 'estimator_mediated',
    reachable                  INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (projection_kind, projection_id)
);
CREATE INDEX IF NOT EXISTS idx_projection_basis_source
    ON projection_basis(source_entity_id);

-- #874 Activation-gated sparse writes: the reviewable write-quarantine hold.
-- A write whose measured interference exceeds the configured bound is
-- staged here (body encrypted like entities when encryption is on) instead
-- of committing to memory — never served by any read surface, listed via
-- `perseus_vault_write_quarantine` for operator review (release materializes
-- the write through the audited remember path; delete drops it). The full
-- interference report rides in `interference_json`; the journal carries the
-- audit trail (`interference_quarantined` / `interference_released` /
-- `interference_deleted`).
CREATE TABLE IF NOT EXISTS write_quarantine (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    body_json TEXT NOT NULL,
    links TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    entity_type TEXT NOT NULL DEFAULT 'insight',
    importance REAL NOT NULL DEFAULT 0.5,
    certainty REAL NOT NULL DEFAULT 0.5,
    visibility TEXT NOT NULL DEFAULT 'workspace',
    topic_path TEXT NOT NULL DEFAULT '',
    always_on INTEGER NOT NULL DEFAULT 0,
    agent_id TEXT NOT NULL DEFAULT '',
    workspace_hash TEXT NOT NULL DEFAULT '',
    valid_from_unix_ms INTEGER,
    valid_to_unix_ms INTEGER,
    interference_score REAL NOT NULL,
    interference_json TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT 'interference',
    created_at_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_write_quarantine_ws
 ON write_quarantine(workspace_hash, created_at_unix_ms);

-- #1026 admission quarantine disposition: sealed REJECTED candidates.
-- When trust admission disposes a candidate as `quarantined`, the sealed
-- payload + attempt metadata land HERE, OUTSIDE the authoritative head —
-- storage presence confers no authority, and no read surface serves it.
-- Reachable only through `perseus_vault_admission_quarantine` (operator
-- review). `proposal_id` is a caller-supplied attempt identifier: once a
-- workspace disposes a proposal_id as quarantined, reusing it is refused
-- with a stable RetiredIdentifier-equivalent error (row retained until
-- purged). `admission_decision_digest` links the attempt outcome; the
-- `receipt_digest` is the canonical hash-only receipt of the sealed
-- record. Body is encrypted like entities when encryption is on.
CREATE TABLE IF NOT EXISTS admission_quarantine (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    body_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    workspace_hash TEXT NOT NULL DEFAULT '',
    agent_id TEXT NOT NULL DEFAULT '',
    actor_kind TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT 'quarantined',
    reason_codes TEXT NOT NULL DEFAULT '[]',
    record_digest TEXT NOT NULL DEFAULT '',
    admission_decision_digest TEXT NOT NULL DEFAULT '',
    receipt_digest TEXT NOT NULL DEFAULT '',
    decay_score REAL NOT NULL DEFAULT 0.5,
    created_at_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_admission_quarantine_ws
 ON admission_quarantine(workspace_hash, created_at_unix_ms);
CREATE UNIQUE INDEX IF NOT EXISTS idx_admission_quarantine_proposal
 ON admission_quarantine(workspace_hash, proposal_id)
 WHERE proposal_id != '';

-- #1027 epoch-fenced writer handoff: the per-workspace writer directory.
-- pointer_state machine: '' (unfenced) -> prepared -> fenced -> active.
-- Fence CLEARS the writer and advances the epoch: after a fence NO writer
-- is authorized (a crash in the Fence->Activate gap leaves zero writers,
-- fail-closed, never two). Activate runs admission against the exact fenced
-- revision: the activating agent must equal the current target AND present
-- the current epoch. Every write against a directory row must carry a
-- matching writer_epoch; stale epochs fail with a stable StaleRevision /
-- WriterEpoch reason. lifecycle_json is the append-only signed receipt log
-- (canonical digests per lifecycle result).
CREATE TABLE IF NOT EXISTS writer_directory (
    workspace_hash TEXT PRIMARY KEY,
    epoch INTEGER NOT NULL DEFAULT 0,
    pointer_state TEXT NOT NULL DEFAULT '',
    writer_agent_id TEXT NOT NULL DEFAULT '',
    target_agent_id TEXT NOT NULL DEFAULT '',
    lifecycle_json TEXT NOT NULL DEFAULT '[]',
    updated_at_unix_ms INTEGER NOT NULL
);

-- #871: durable long-running operation states — shared run/run-item contract
-- for maintenance, embed, consolidation, export/import, and reindex
-- operations. Terminal states: completed | failed | cancelled | interrupted |
-- failed_to_start; orthogonal partial/timeout/stale flags; per-item receipts
-- for fan-out isolation and bounded scoped retry.
CREATE TABLE IF NOT EXISTS op_runs (
    id TEXT PRIMARY KEY,
    op_type TEXT NOT NULL,
    state TEXT NOT NULL,
    partial INTEGER NOT NULL DEFAULT 0,
    timeout INTEGER NOT NULL DEFAULT 0,
    stale INTEGER NOT NULL DEFAULT 0,
    scope TEXT NOT NULL DEFAULT '',
    input_digest TEXT NOT NULL DEFAULT '',
    items_total INTEGER NOT NULL DEFAULT 0,
    items_done INTEGER NOT NULL DEFAULT 0,
    items_failed INTEGER NOT NULL DEFAULT 0,
    items_unattempted INTEGER NOT NULL DEFAULT 0,
    error_class TEXT NOT NULL DEFAULT '',
    error_detail TEXT NOT NULL DEFAULT '',
    receipt TEXT NOT NULL DEFAULT '',
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 2,
    parent_run_id TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL DEFAULT '',
    created_at_unix_ms INTEGER NOT NULL,
    started_at_unix_ms INTEGER,
    updated_at_unix_ms INTEGER NOT NULL,
    finished_at_unix_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_op_runs_state
 ON op_runs(state, updated_at_unix_ms);
CREATE INDEX IF NOT EXISTS idx_op_runs_type
 ON op_runs(op_type, created_at_unix_ms);
CREATE TABLE IF NOT EXISTS op_run_items (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    item_ref TEXT NOT NULL,
    item_digest TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,
    receipt_ref TEXT NOT NULL DEFAULT '',
    error_class TEXT NOT NULL DEFAULT '',
    error_detail TEXT NOT NULL DEFAULT '',
    retry_count INTEGER NOT NULL DEFAULT 0,
    created_at_unix_ms INTEGER NOT NULL,
    updated_at_unix_ms INTEGER NOT NULL,
    finished_at_unix_ms INTEGER,
    UNIQUE(run_id, item_ref)
);
CREATE INDEX IF NOT EXISTS idx_op_run_items_run
 ON op_run_items(run_id, state);

-- ── v36 (#875): learned anticipation — preload usage telemetry + tuning ──
-- preload_events: one row per preloaded item per serve (recall_when result,
-- context-block injection). `used` stays NULL until resolution (touch-after-
-- serve check); trigger_ref is the matched trigger string or a sentinel
-- (__always_on__ / __keyword__ / __context_block__).
CREATE TABLE IF NOT EXISTS preload_events (
    id TEXT PRIMARY KEY,
    ts INTEGER NOT NULL,
    context_hash TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    entity_id TEXT NOT NULL,
    trigger_ref TEXT NOT NULL,
    workspace_hash TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    la_before INTEGER NOT NULL,
    used INTEGER,
    resolved_ts INTEGER
);
CREATE INDEX IF NOT EXISTS idx_preload_events_resolve
 ON preload_events(used, ts);
CREATE INDEX IF NOT EXISTS idx_preload_events_entity
 ON preload_events(entity_id, ts);

-- preload_sessions: per-session resolution rows (session_id, or a
-- pseudo-session per context_hash when the caller passed none).
-- context_words: meaning-bearing words of the session contexts (union);
-- missed_by_trigger_json: map trigger string -> count of missed entities
-- whose own recall_when matched the session context but were not served.
CREATE TABLE IF NOT EXISTS preload_sessions (
    session_id TEXT PRIMARY KEY,
    anchor_ts INTEGER NOT NULL,
    preloaded_n INTEGER NOT NULL,
    used_n INTEGER NOT NULL,
    missed_n INTEGER NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    miss_rate REAL NOT NULL,
    context_words TEXT NOT NULL DEFAULT '[]',
    missed_by_trigger_json TEXT NOT NULL DEFAULT '{}',
    missed_ids_json TEXT NOT NULL DEFAULT '[]',
    resolved_ts INTEGER NOT NULL
);

-- preload_proposals: operator review queue for trigger tuning. Mutations
-- apply ONLY through review approve (journal + audited remember path).
CREATE TABLE IF NOT EXISTS preload_proposals (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    trigger_ref TEXT NOT NULL,
    suggestion TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,
    created_ts INTEGER NOT NULL,
    decided_ts INTEGER,
    decided_by TEXT NOT NULL DEFAULT '',
    applied_ts INTEGER,
    journal_event_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_preload_proposals_state
 ON preload_proposals(state, created_ts);
";

/// Current schema migration level, stamped into `PRAGMA user_version` once all
/// the column-add migrations below have been applied. Bump this whenever you add
/// a new ALTER-probe migration in `initialize_schema`, or existing databases
/// (already at the previous level) will skip it.
///
/// v16 (#503): #487 added the usefulness columns (`usefulness_count`,
/// `last_useful_unix_ms`) WITHOUT bumping this, so every store already at v15
/// (anything upgraded through v2.18.x) skipped the ALTERs — decay_tick, and
/// with it maintain / --maintain-every / autocohere, failed with "no such
/// column: usefulness_count" (hit live on the first v2.19.0 production
/// deploy). The whole migration function is idempotent (ensure_column probes;
/// the v15 rehash is deterministic and the keyed-chain rekey is canary-gated),
/// so the bump simply re-runs it once and picks up the missing columns.
///
/// v17 (#476): dedup_signatures gains scope columns (category,
/// workspace_hash) + the (category, workspace_hash, tg_count) band index, and
/// every active entity's missing signature is backfilled once — the
/// near-duplicate scan now drives off this small table instead of hydrating
/// (and re-hashing) every same-category entity body per write, which was
/// O(N·body_size) per insert and the cause of the #474-measured quadratic
/// bulk-load curve (141/s @10K → 7/s @100K).
///
/// v18 (#507): covering partial index for dense search's phase-0 signature
/// scan and embedded-row count. `embedding`/`emb_sig` are late ALTER columns
/// stored AFTER body_json in each record, so "read id + emb_sig for every
/// embedded row" walked every row's multi-KB overflow chain — ~900MB of page
/// reads per dense query at 100K (390ms p50). The index is keyed on
/// `emb_sig IS NOT NULL` (all its columns are index columns, so the residual
/// predicate evaluates from the index and the scan covers on the bundled
/// SQLite; the `embedding IS NOT NULL` spelling never covers, and an
/// expression-index variant only covers on SQLite ≥3.5x). The migration
/// backfills emb_sig from every stored embedding (a pure function of the
/// vector), making "embedded ⟺ signed" an invariant — writers already set
/// and clear both columns together.
/// v19 (#619 step 2b): `emb_sig4` — 4-bit scalar-quantized embedding codes
/// (per-row scale + nibbles) for the int4 ADC refine tier between the 1-bit
/// coarse prefilter and the exact rerank, plus its covering index. Backfilled
/// from stored vectors at migration; writers maintain the column alongside
/// embedding/emb_sig.
/// v20 (#682 Temporal RAG): `entity_history_fts` — standalone FTS5 over
/// superseded/retired body terms, so point-in-time semantic recall can surface
/// facts whose query-matching version has since left the live index. Created
/// idempotently; mode-aware reindexing backfills existing rows when the key is
/// available, using plaintext terms for plaintext stores and keyed blind tokens
/// for protected stores.
/// v21 (#683 Keystones): the `keystones` table (mandatory policy rules).
/// Created idempotently; no backfill (new table).
/// v22 (#684 Multi-agent scoping): the `agents` registry (trust_tier + fleet).
/// Created idempotently; no backfill (new table). entities/journal already
/// carry agent_id from v1.2.0.
/// v23 (Chancery audit cross-referencing, #6): `chancery_writ_id` column on
/// the journal table. When Chancery wraps an MCP server, it stamps every
/// tools/call with `_meta.chancery/lease` — the writ's unique ID. This column
/// records it so downstream audit queries can cross-reference vault journal
/// entries against Chancery writ records. New column, backfill-free: no-op on
/// fresh DBs; legacy rows get '' and are populated going forward.
/// v24 (#768 Authorized Action Receipts): authority manifests, receipts, and
/// action leases. Tables are created idempotently; action columns are ALTER-
/// probed because v23 stores must retain existing receipt records.
/// v25 (#811 Immutable artifacts): shared content-addressed `artifacts` bytes
/// plus scope/provenance/representation `artifact_bindings`.
/// v27 (#880 epistemic states): `epistemic_state` trust axis on entities —
/// 'candidate' | 'verified' | 'corroborated' | 'rejected' |
/// 'defensively_recalled', orthogonal to the lifecycle `status` column.
/// Backfill-free: existing rows default to 'candidate' (useful but unverified),
/// which is the safe reading for any legacy record lacking admission evidence.
/// v33 (#885 vector compression): `embedding_format` (store-wide declared
/// `entities.embedding` storage format: float32 | int8 | bit) and
/// `entities_embedding_snapshot` (pre-quantization float32 column backup for
/// the documented rollback path). New tables, idempotent, no backfill — the
/// format record is written by the reindex path / fresh-store open, and the
/// snapshot only by the quantization step.
/// v34 (#874 activation-gated sparse writes): `write_quarantine` — the
/// reviewable hold for writes whose measured interference exceeds the
/// configured bound. New table, idempotent, no backfill.
/// v39 (#990 deletion-residue accounting): `projection_basis` — declared
/// basis for tier-3 projections (embeddings first). New table, idempotent,
/// no backfill (DDL_V0_2_0 is re-run at every open, so existing stores pick
/// it up on next open).
/// v43 (#1020 fingerprint tier): `entities.fingerprint` — deterministic
/// subword-HDC sign-bit fingerprint of the plaintext body, written on
/// content change only while the tier is enabled, NULL otherwise. Additive,
/// no backfill (pre-enablement rows simply have no fingerprint and stay
/// out of the zero-API fallback pool).
/// v44 (#1026 admission quarantine disposition): `admission_quarantine` —
/// sealed rejected candidates retained outside the authoritative head,
/// linked to their attempt outcome (admission decision digest) and hash-only
/// receipt, reachable only through explicit review tooling. New table,
/// idempotent, no backfill.
/// v45 (#1027 epoch-fenced writer handoff): `writer_directory` — per-workspace
/// epoch-fenced single-writer handoff (Prepare/Fence/Retarget/Activate with
/// monotonically advanced epochs; Fence clears the writer so a crash in the
/// Fence→Activate gap leaves zero writers, fail-closed). Writes against an
/// active directory must carry a matching `writer_epoch`. New table,
/// idempotent, no backfill.
/// v46 (#1029 supersession impact index): `authorized_actions.justification_json`
///
/// v47 (#1033 compensation admission): `impact_findings` table (durable
/// detection records) + four `authorized_actions` compensation-linkage
/// columns (`compensates_for`, `finding_ref`, `superseding_head`,
/// `handoff_receipt_ref`) — compensation intents must cite an authenticated
/// finding + superseding head; self-claimed undo is rejected fail-closed.
///
/// v48 (#1034 grounding verification): `grounding_fingerprints` table —
/// deterministic content fingerprints (K=64 seeded-sha256 trigram MinHash +
/// neighbor set) captured at admission for evidence grounded to
/// files/symbols; maintenance passes reconcile current content vs the
/// baseline (ok / drift / moved / gone / ambiguous) with auto-rewrite +
/// provenance trail on MOVED.
/// — the entity ids an action cited as grounding, so the reverse impact
/// closure can flag PENDING actions whose justification changed. Additive
/// column, no backfill (pre-existing actions cite nothing).
/// v59 (#1173 non-authoritative experience projections): a scoped, compact
/// projection row plus normalized canonical-source links and a rebuild ledger.
/// These tables contain only IDs, digests, bounded signals, and explicit scope;
/// they never become an answer-facing source of truth.
/// v60 (#1182 task-scoped serving state): a scoped, compact, rebuildable
/// projection containing task metadata, canonical evidence references, and
/// digests only. It is not canonical memory/history and never stores prompts,
/// model reasoning, or entity bodies.
pub(crate) const SCHEMA_VERSION: i64 = 60;

/// Initialize the v0.2.0 schema on a fresh database.
pub fn initialize_schema(conn: &Connection) -> Result<(), Box<dyn std::error::Error>> {
    // #353: serialize the ENTIRE schema bootstrap, not only ALTER migrations.
    // `DDL_V0_2_0` is idempotent but is still write DDL (including FTS5 virtual
    // tables). Two fresh/pre-upgrade openers could race there before the old
    // BEGIN IMMEDIATE below, producing SQLITE_BUSY despite the migration lock.
    // SQLite DDL, including the virtual-table creation used here, is
    // transactional; acquire the cross-process write mutex before any schema
    // statement so the loser waits, then observes the winner's complete state.
    conn.execute_batch("BEGIN IMMEDIATE;")?;
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        conn.execute_batch(DDL_V0_2_0)?;

        // `open` runs several times per process, so fully-migrated stores do
        // no column probes after the in-transaction version check. A process
        // that waited on the lock sees the winner's stamped version here.
        let user_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if user_version > SCHEMA_VERSION {
            return Err(format!(
                "unsupported future schema version {user_version}; this binary supports {SCHEMA_VERSION}"
            )
            .into());
        }
        if user_version < SCHEMA_VERSION {
            apply_migrations(conn)?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => {
            conn.execute_batch("COMMIT;")?;
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK;");
            Err(e)
        }
    }
}

/// Add `column` to `table` unless it already exists. Defense-in-depth for the
/// #353 race: even if the existence probe raced another writer (e.g. a process
/// that migrated between our probe and our ALTER), "duplicate column name"
/// means the column is present — exactly the state we wanted — so it is
/// treated as success, not an error.
fn ensure_column(
    conn: &Connection,
    table: &str,
    column: &str,
    decl: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if conn
        .prepare(&format!("SELECT {column} FROM {table} LIMIT 1"))
        .is_ok()
    {
        return Ok(());
    }
    match conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl};")) {
        Ok(()) => Ok(()),
        Err(e) if e.to_string().contains("duplicate column name") => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// The gated column-add migrations. Runs inside the BEGIN IMMEDIATE
/// transaction taken by `initialize_schema` (#353) — must not BEGIN/COMMIT
/// itself.
fn apply_migrations(conn: &Connection) -> Result<(), Box<dyn std::error::Error>> {
    // Re-check the version now that we hold the write lock: if another
    // process completed the migration while we waited on BEGIN IMMEDIATE,
    // there is nothing left to do. (#353)
    let user_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if user_version >= SCHEMA_VERSION {
        return Ok(());
    }

    // Add embedding column if it doesn't exist (migration from v0.2.0)
    ensure_column(conn, "entities", "embedding", "BLOB")?;

    // v1.x: always_on, certainty
    ensure_column(conn, "entities", "always_on", "INTEGER DEFAULT 0")?;
    ensure_column(conn, "entities", "certainty", "REAL DEFAULT 0.5")?;

    // v1.2.0: multi-workspace scoping, agent attribution, access controls
    ensure_column(conn, "entities", "workspace_hash", "TEXT DEFAULT ''")?;
    ensure_column(conn, "entities", "agent_id", "TEXT DEFAULT ''")?;
    ensure_column(conn, "journal", "agent_id", "TEXT DEFAULT ''")?;
    ensure_column(conn, "entities", "visibility", "TEXT DEFAULT 'workspace'")?;

    // v2.0: cryptographic audit log
    ensure_column(conn, "journal", "audit_hash", "TEXT DEFAULT ''")?;

    // Add bi-temporal columns (v2.4.0 — bi-temporal facts). Valid time
    // (valid_from/valid_to), transaction time (recorded_at/invalidated_at), and
    // supersession links. All additive; existing rows keep their meaning.
    ensure_column(conn, "entities", "valid_from_unix_ms", "INTEGER")?;
    ensure_column(conn, "entities", "valid_to_unix_ms", "INTEGER")?;
    ensure_column(conn, "entities", "recorded_at_unix_ms", "INTEGER")?;
    ensure_column(conn, "entities", "invalidated_at_unix_ms", "INTEGER")?;
    ensure_column(conn, "entities", "supersedes", "TEXT DEFAULT ''")?;
    ensure_column(conn, "entities", "superseded_by", "TEXT DEFAULT ''")?;

    // Add efficacy-tracking columns (v2.10.0 — PMB-inspired follow-rate scoring).
    ensure_column(conn, "entities", "follow_count", "INTEGER DEFAULT 0")?;
    ensure_column(conn, "entities", "miss_count", "INTEGER DEFAULT 0")?;
    ensure_column(conn, "entities", "follow_rate", "REAL DEFAULT 0.0")?;
    ensure_column(
        conn,
        "entities",
        "efficacy_status",
        "TEXT DEFAULT 'unverified'",
    )?;

    // Add usefulness-tracking columns (#487 — derived_from reinforcement).
    // v16 (#503): these shipped without a SCHEMA_VERSION bump, so v15 stores
    // never ran them — the bump to 16 exists to deliver exactly these two
    // ALTERs to already-migrated stores.
    ensure_column(conn, "entities", "usefulness_count", "INTEGER DEFAULT 0")?;
    ensure_column(conn, "entities", "last_useful_unix_ms", "INTEGER DEFAULT 0")?;

    // v28 (#868): retention expiry on the live row. NULL = never expires
    // (the correct reading for every legacy row), so this is purely additive.
    ensure_column(conn, "entities", "expires_at_unix_ms", "INTEGER")?;
    ensure_column(
        conn,
        "entities",
        "preload_tuned_unix_ms",
        "INTEGER DEFAULT 0",
    )?;

    // #1000: typed memory classes (CogniCore borrow). '' = legacy row
    // (SEMANTIC policy, byte-compatible with pre-#1000 behavior); validated
    // at write time against the MemoryType taxonomy. Additive.
    ensure_column(conn, "entities", "memory_type", "TEXT DEFAULT ''")?;

    // #1001: utility-driven promotion — the accrued-usage signal that
    //    feeds the pure candidate→verified transition. Saturating cap
    //    enforced at every bump. Additive; existing rows start at 0.
    ensure_column(conn, "entities", "utility_score", "REAL NOT NULL DEFAULT 0")?;

    // #1020: deterministic subword-HDC fingerprint tier. Additive and
    //    backfill-free: rows written before enablement simply have NULL
    //    and stay out of the zero-API fallback pool; enablement covers
    //    new and changed bodies from the next write onward.
    //    DELIBERATELY absent from DDL_V0_2_0: ALTER appends at the end, and
    //    the DDL must keep the exact physical column order migrated stores
    //    get, because `entity_from_row` hydrates `SELECT *` projections
    //    positionally (e.g. the intention readers in tools.rs). A column
    //    placed mid-DDL would shift every later column on FRESH stores only.
    ensure_column(conn, "entities", "fingerprint", "BLOB")?;

    // #1026: admission quarantine disposition — sealed rejected candidates
    //    live OUTSIDE the authoritative head. New table only; nothing
    //    existed before to migrate (previous quarantined writes landed in
    //    `entities` and stay there as historical non-authoritative rows).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS admission_quarantine (
            id TEXT PRIMARY KEY,
            proposal_id TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            body_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            workspace_hash TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT '',
            actor_kind TEXT NOT NULL DEFAULT '',
            outcome TEXT NOT NULL DEFAULT 'quarantined',
            reason_codes TEXT NOT NULL DEFAULT '[]',
            record_digest TEXT NOT NULL DEFAULT '',
            admission_decision_digest TEXT NOT NULL DEFAULT '',
            receipt_digest TEXT NOT NULL DEFAULT '',
            decay_score REAL NOT NULL DEFAULT 0.5,
            created_at_unix_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_admission_quarantine_ws
          ON admission_quarantine(workspace_hash, created_at_unix_ms);
         CREATE UNIQUE INDEX IF NOT EXISTS idx_admission_quarantine_proposal
          ON admission_quarantine(workspace_hash, proposal_id)
          WHERE proposal_id != '';",
    )?;

    // #1027: epoch-fenced writer handoff — the per-workspace writer
    //    directory. New table only; no backfill (no handoff state existed
    //    before; an absent row means unfenced, which is the legacy posture).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS writer_directory (
            workspace_hash TEXT PRIMARY KEY,
            epoch INTEGER NOT NULL DEFAULT 0,
            pointer_state TEXT NOT NULL DEFAULT '',
            writer_agent_id TEXT NOT NULL DEFAULT '',
            target_agent_id TEXT NOT NULL DEFAULT '',
            lifecycle_json TEXT NOT NULL DEFAULT '[]',
            updated_at_unix_ms INTEGER NOT NULL
         );",
    )?;

    // #1029: supersession impact index — actions record the entities they
    //    cited as grounding, so a later supersede/retract can enumerate
    //    which pending actions must re-validate their justification.
    //    Additive column; index-based hydration only (no SELECT *), and
    //    ALTER appends at the end of the physical row, so the fresh-DDL
    //    column order stays identical to migrated stores.
    ensure_column(
        conn,
        "authorized_actions",
        "justification_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )?;

    // #1033: compensation admission — compensation/undo intents must cite an
    //    authenticated impact finding + the superseding head that invalidated
    //    the original justification. Additive columns, ALTER-appended at the
    //    physical end of the row (index-based hydration only), so fresh-DDL
    //    column order stays identical to migrated stores. `impact_findings`
    //    is a new table (durable detection records); nothing existed before.
    ensure_column(
        conn,
        "authorized_actions",
        "compensates_for",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "authorized_actions",
        "finding_ref",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "authorized_actions",
        "superseding_head",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "authorized_actions",
        "handoff_receipt_ref",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS impact_findings (
            id TEXT PRIMARY KEY,
            finding_ref TEXT NOT NULL,
            workspace_hash TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL DEFAULT '',
            key TEXT NOT NULL DEFAULT '',
            entity_id TEXT NOT NULL DEFAULT '',
            cited_head TEXT NOT NULL,
            covers_json TEXT NOT NULL DEFAULT '[]',
            basis TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'open',
            archived INTEGER NOT NULL DEFAULT 0,
            created_at_unix_ms INTEGER NOT NULL
         );
         CREATE UNIQUE INDEX IF NOT EXISTS idx_impact_findings_ref
          ON impact_findings(workspace_hash, finding_ref);
         CREATE INDEX IF NOT EXISTS idx_impact_findings_ws
          ON impact_findings(workspace_hash, status, archived);",
    )?;

    // #1034: grounding verification — deterministic content fingerprints for
    //    evidence grounded to files/symbols. New table only; nothing existed
    //    before. `status` transitions ok -> drift/moved/gone/ambiguous on
    //    reconcile passes; `provenance_json` is the append-only migration
    //    trail (MOVED never silent last-write-wins).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS grounding_fingerprints (
            id TEXT PRIMARY KEY,
            workspace_hash TEXT NOT NULL DEFAULT '',
            entity_id TEXT NOT NULL,
            target_ref TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'file',
            fingerprint_hex TEXT NOT NULL DEFAULT '',
            neighbor_count INTEGER NOT NULL DEFAULT 0,
            neighbors_json TEXT NOT NULL DEFAULT '[]',
            baseline_digest TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'ok',
            candidates_json TEXT NOT NULL DEFAULT '[]',
            provenance_json TEXT NOT NULL DEFAULT '[]',
            reviewed_at_unix_ms INTEGER,
            captured_at_unix_ms INTEGER NOT NULL,
            updated_at_unix_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_grounding_fp_ws
          ON grounding_fingerprints(workspace_hash, status);
         CREATE INDEX IF NOT EXISTS idx_grounding_fp_entity
          ON grounding_fingerprints(entity_id);",
    )?;

    // v29 (#876): governed-distillation lifecycle on artifact bindings.
    // A learned artifact (trained weights / distilled cartridge) is bound to
    // its source entities at registration; when a source is physically
    // erased the binding is REVOKED (serve paths refuse it), and when a
    // source is superseded it is flagged STALE (retraining trigger, journal
    // evidence). Both are additive flags; NULL = live binding.
    ensure_column(conn, "artifact_bindings", "revoked_at_unix_ms", "INTEGER")?;
    ensure_column(conn, "artifact_bindings", "stale_at_unix_ms", "INTEGER")?;
    ensure_column(
        conn,
        "artifact_bindings",
        "revocation_reason",
        "TEXT DEFAULT ''",
    )?;

    // v29 (#876): source-entity bindings for learned artifacts. One row per
    // (artifact binding, source entity) with hash-only evidence (entity id +
    // normalized body digest + recorded_at), so revocation scans and receipt
    // replay can bind artifact -> sources -> workspace. CASCADE keeps the
    // table in step with artifact_bindings deletes (pool opens with
    // foreign_keys=ON).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS learned_artifact_sources (
            binding_id TEXT NOT NULL REFERENCES artifact_bindings(binding_id) ON DELETE CASCADE,
            entity_id TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT '',
            key TEXT NOT NULL DEFAULT '',
            value_sha256 TEXT NOT NULL,
            recorded_at_unix_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_learned_sources_entity
            ON learned_artifact_sources(entity_id);",
    )?;

    // v30 (#879): first-class Hermes profile <-> Vault workspace binding.
    // One row per profile (PK); a workspace may be shared by several
    // profiles (intentional shared memory). access_mode enforces read-only
    // vs read/write at the tool boundary; binding_state drives lifecycle
    // controls (active | quarantined | unbound); last_seen_unix_ms is the
    // client heartbeat so stale bindings are diagnosable. Bindings are
    // journaled (workspace_bound/rebound/unbound/quarantined/reactivated).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS workspace_bindings (
            profile_name TEXT PRIMARY KEY,
            workspace_hash TEXT NOT NULL,
            access_mode TEXT NOT NULL DEFAULT 'read_write',
            binding_state TEXT NOT NULL DEFAULT 'active',
            quarantine_reason TEXT NOT NULL DEFAULT '',
            bound_at_unix_ms INTEGER NOT NULL,
            rebound_at_unix_ms INTEGER,
            unbound_at_unix_ms INTEGER,
            last_seen_unix_ms INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}'
         );
         CREATE INDEX IF NOT EXISTS idx_workspace_bindings_ws
            ON workspace_bindings(workspace_hash);",
    )?;

    // Backfill transaction time for pre-existing rows: a fact's recorded_at is
    // when Perseus Vault first stored it, i.e. its created_at. (No-op on a fresh DB.)
    conn.execute_batch(
        "UPDATE entities SET recorded_at_unix_ms = created_at_unix_ms \
         WHERE recorded_at_unix_ms IS NULL;",
    )?;

    // Live-fact filter index. Created here (not in the ungated DDL) because it
    // references invalidated_at_unix_ms, which on a migrating DB only exists
    // after the ALTER above. NULL = live; recall will exclude non-NULL rows.
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_entities_invalidated \
         ON entities(invalidated_at_unix_ms);",
    )?;

    // v5: persistent importance floor (see the column comment in the DDL).
    ensure_column(conn, "entities", "importance", "REAL DEFAULT 0.0")?;

    // v6: sign-bit embedding signatures for the dense-search prefilter, plus a
    // backfill for embeddings stored before the column existed. Bounded work:
    // one pass over embedded rows, ~50 bytes written per row.
    ensure_column(conn, "entities", "emb_sig", "BLOB")?;
    {
        let mut stmt = conn.prepare(
            "SELECT id, embedding FROM entities \
             WHERE embedding IS NOT NULL AND emb_sig IS NULL",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?;
        let pending: Vec<(String, Vec<u8>)> = rows.filter_map(|r| r.ok()).collect();
        drop(stmt);
        for (id, blob) in pending {
            let emb: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let sig = crate::db::embedding_signature(&emb);
            conn.execute(
                "UPDATE entities SET emb_sig = ?1 WHERE id = ?2",
                params![sig, id],
            )?;
        }
    }

    // v4 (#339): identity becomes (category, key, workspace_hash). A plain
    // (category, key) uniqueness made cross-workspace key collisions
    // unstorable, which is what forced perseus_vault_share's "copy into workspace" to
    // clobber the source row. Created here (after the workspace_hash ALTER,
    // like idx_entities_invalidated) rather than in the ungated DDL. Safe on
    // a populated DB — the old constraint was strictly tighter, so no
    // existing rows can collide. Create-then-drop, so uniqueness is never
    // unenforced.
    conn.execute_batch(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_category_key_ws \
         ON entities(category, key, workspace_hash); \
         DROP INDEX IF EXISTS idx_entities_category_key;",
    )?;

    // ── v8 (#365): GraphRAG communities ─────────────────────────────────
    // Self-contained block (easy to renumber at merge). Persists the output
    // of community detection over the entity link graph: one row per detected
    // community, scoped to a workspace. `id` is derived from a digest of the
    // sorted member ids, so a membership change produces a NEW community id —
    // that is the cache-invalidation mechanism for community summaries (the
    // state-digest cache-key pattern from #256). All DDL is IF NOT EXISTS,
    // so this block is idempotent and safe to re-run.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS communities (
            id TEXT PRIMARY KEY,                        -- 'com-' + digest of sorted member ids
            workspace_hash TEXT NOT NULL DEFAULT '',
            member_ids TEXT NOT NULL DEFAULT '[]',      -- JSON array of entity ids
            member_digest TEXT NOT NULL DEFAULT '',     -- digest of the member set (cache key)
            summary TEXT NOT NULL DEFAULT '',           -- extractive (or LLM-polished) summary
            summary_entity_id TEXT NOT NULL DEFAULT '', -- entities.id of the stored summary entity
            algorithm TEXT NOT NULL DEFAULT 'label_prop',
            modularity REAL NOT NULL DEFAULT 0.0,       -- partition modularity of the detection run
            member_count INTEGER NOT NULL DEFAULT 0,
            generated_at_unix_ms INTEGER NOT NULL DEFAULT 0
         );
         CREATE INDEX IF NOT EXISTS idx_communities_ws ON communities(workspace_hash);",
    )?;

    // ── v9 (#363): bi-temporal valid-time backfill ──────────────────────
    // Self-contained block (renumber-safe). The valid-time axis becomes
    // queryable (perseus_vault_valid_at / perseus_vault_bitemporal / recall valid filters),
    // so make the historical convention "NULL valid_from = valid since the
    // fact was recorded" explicit: backfill valid_from to the row's
    // transaction time. valid_to stays NULL (= still true / unbounded).
    // Idempotent and re-runnable — only NULL rows are touched, and query
    // paths still COALESCE for rows written by older binaries afterwards.
    conn.execute_batch(
        "UPDATE entities SET valid_from_unix_ms = COALESCE(recorded_at_unix_ms, created_at_unix_ms) \
         WHERE valid_from_unix_ms IS NULL; \
         UPDATE entity_history SET valid_from_unix_ms = COALESCE(recorded_at_unix_ms, created_at_unix_ms) \
         WHERE valid_from_unix_ms IS NULL;",
    )?;
    // ── end v9 ──────────────────────────────────────────────────────────

    // ── v10 (#392): stored near-duplicate signatures ─────────────────────
    // Self-contained block (renumber-safe). One row per entity holding the
    // packed character-trigram set of the STORED body_json column value (see
    // src/dedup.rs), so find_near_duplicate can compute its exact Jaccard
    // verdict without rebuilding the trigram set per candidate per insert —
    // the O(M·N) cost behind the 1.6s-per-write stall at 50k rows.
    //
    // Backfill is LAZY: rows written before this migration simply have no
    // signature; the dedup scan takes the old rebuild-from-body path for
    // them (identical verdicts, old cost) and writes the signature back in
    // bounded batches, so a large store converges without a potentially
    // multi-minute eager migration. body_len records the stored body's byte
    // length as a freshness guard. ON DELETE CASCADE keeps the side table in
    // step with every entity-delete path (the pool opens connections with
    // foreign_keys=ON); a surviving orphan is inert — the scan joins FROM
    // entities, so it can never resurrect a deleted row.
    // Two tables, split by access pattern. The scan LEFT JOINs
    // dedup_signatures (small fixed-size rows: freshness guard, set size,
    // 256-byte prune histogram) for EVERY candidate, so its per-row page
    // footprint must stay tiny — a writer committing between scans
    // invalidates other pooled connections' page caches, making the scan's
    // hot page count the real cost on write-heavy workloads. The multi-KB
    // exact trigram set lives in dedup_signature_blobs and is fetched by a
    // separate point query ONLY for the rare candidate that survives both
    // lossless prunes. WITHOUT ROWID: the entity_id probes hit one clustered
    // b-tree instead of autoindex-then-rowid double lookups.
    //
    // Freshness guard = (body_len, body_hash): a signature is trusted only
    // while BOTH match the stored body. Length alone cannot catch a
    // same-length rewrite by a signature-unaware writer — which is exactly
    // what a rolled-back pre-v10 binary produces (it rewrites body_json
    // without touching these tables; AES-GCM re-encryption even preserves
    // ciphertext length). With the hash, such rows read as stale, fall back
    // to the exact rebuild path (identical verdicts), and self-heal on the
    // next dedup touch — so v10 stores stay ROLLBACK-SAFE: running an older
    // binary never poisons dedup verdicts, and dropping both side tables is
    // always a safe reset (they hold only derived, rebuildable data).
    // category/workspace_hash (v17, #476): the scan is signature-driven and
    // must filter scope WITHOUT joining entities (that join is what hydrated
    // every multi-KB body per write). Fresh DBs get the full shape here;
    // pre-v17 stores get the columns + backfill in the gated v17 block.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS dedup_signatures (
            entity_id TEXT PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
            body_len INTEGER NOT NULL,
            body_hash INTEGER NOT NULL,
            tg_count INTEGER NOT NULL,
            histo BLOB,
            category TEXT NOT NULL DEFAULT '',
            workspace_hash TEXT NOT NULL DEFAULT ''
         ) WITHOUT ROWID;
         CREATE TABLE IF NOT EXISTS dedup_signature_blobs (
            entity_id TEXT PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
            sig BLOB NOT NULL
         ) WITHOUT ROWID;",
    )?;
    // ── end v10 ─────────────────────────────────────────────────────────

    // ── v11: per-workspace journal scoping (#417) ───────────────────────
    // journal had no workspace column, so purge's (category, key) redaction
    // over-redacted cross-workspace same-key rows. Add the column + its match
    // index. Legacy rows keep '' (unknown) and stay conservatively matched by
    // purge; new rows are stamped at write time in Database::journal.
    // category/key have been in the journal table since v0.2.0 and are already
    // assumed by purge's JRN_MATCH and the journal readers; ensure_column them
    // here so the composite index below is robust even on ancient pre-v0.2.0
    // journals (and minimal test fixtures) that predate those columns.
    ensure_column(conn, "journal", "category", "TEXT DEFAULT ''")?;
    ensure_column(conn, "journal", "key", "TEXT DEFAULT ''")?;
    ensure_column(
        conn,
        "journal",
        "workspace_hash",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_journal_catkeyws ON journal(category, key, workspace_hash);",
    )?;
    // ── end v11 ─────────────────────────────────────────────────────────

    // ── v12 (#433 M2): bind workspace into the audit-chain hash ──────────
    // Pre-v12 chains hashed only (prev_hash, id, created_at_unix_ms), so a
    // journal entry could be moved between workspaces without breaking the
    // chain. The hash now also folds in workspace_hash (stamped since v11).
    // (The chain rehash that was here is now performed once in the v15 block —
    // the v15 formula supersedes v12's, and the migrate function runs every block
    // top-to-bottom for any DB below SCHEMA_VERSION, so a single final rehash is
    // sufficient. Doing it here as well would fail on legacy journals that don't
    // yet have the payload columns the v15 rehash reads.)
    // ── end v12 ──────────────────────────────────────────────────────────

    // ── v13: cover the browse ORDER BY tie-break in idx_entities_recall ──
    // The empty-query browse path orders by
    //   retrieval_count DESC, last_accessed_unix_ms DESC, id ASC
    // but the pre-v13 index covered only the first two keys, so a large
    // tie-group on (retrieval_count, last_accessed) — a cold or bulk-imported
    // store where last_accessed is uniform — forced SQLite to sort the whole
    // group by id to satisfy LIMIT k (O(tie-group); measured ~30ms browse p50
    // at 1M rows). Adding id ASC as a trailing key lets the index satisfy the
    // full ordering, so browse is a pure k-row range scan again. DROP+recreate
    // because the old index lacks the column; safe on a populated DB (the index
    // is derived, and recall/browse fall back to a scan for the brief window).
    conn.execute_batch(
        "DROP INDEX IF EXISTS idx_entities_recall; \
         CREATE INDEX IF NOT EXISTS idx_entities_recall \
           ON entities(archived, retrieval_count DESC, last_accessed_unix_ms DESC, id ASC);",
    )?;
    // ── end v13 ──────────────────────────────────────────────────────────

    // ── v14 (2026-07-05 security review): cryptographic audit-chain hash ──
    // Pre-v14 the journal chain used a 64-bit, non-cryptographic `DefaultHasher`
    // (SipHash) — brute-forceable for targeted collisions. It is now a real
    // SHA-256 over the same erasure-safe identifying tuple (prev, id,
    // created_at, workspace_hash). Recompute existing chains under the new
    // formula so they verify from here forward. Deterministic + idempotent;
    // a no-op on a fresh DB (empty journal). (Rehash deferred to the v15 block —
    // see the v12 note; the v15 formula supersedes this one.)
    // ── end v14 ──────────────────────────────────────────────────────────

    // ── v15 (2026-07-05 security review): payload commitment + keyed chain ──
    // Add the per-entry payload_commitment column and recompute the chain over
    // (prev, id, created_at, workspace, commitment). At migration time no key is
    // available, so the rehash is UNKEYED; `set_encryption` later rekeys it to
    // HMAC once the encryption key is loaded. Backfills commitments for existing
    // rows. Deterministic + idempotent; no-op on a fresh DB. See
    // docs/audit-chain-keyed-mac-design.md.
    // Ensure every column the rehash reads exists — a very old (pre-bitemporal)
    // journal may predate some of them; ensure_column is idempotent.
    ensure_column(conn, "journal", "event_type", "TEXT DEFAULT 'decision'")?;
    ensure_column(conn, "journal", "evaluated_json", "TEXT DEFAULT '{}'")?;
    ensure_column(conn, "journal", "acted_json", "TEXT DEFAULT '{}'")?;
    ensure_column(conn, "journal", "forward_json", "TEXT DEFAULT '{}'")?;
    ensure_column(conn, "journal", "category", "TEXT DEFAULT ''")?;
    ensure_column(conn, "journal", "key", "TEXT DEFAULT ''")?;
    ensure_column(conn, "journal", "entity_id", "TEXT DEFAULT ''")?;
    ensure_column(conn, "journal", "agent_id", "TEXT DEFAULT ''")?;
    ensure_column(conn, "journal", "payload_commitment", "TEXT DEFAULT ''")?;
    crate::db::rehash_audit_chain(conn)?;
    // ── end v15 ──────────────────────────────────────────────────────────

    // ── v17 (#476): signature-driven dedup scan ──────────────────────────
    // The near-duplicate scan used to hydrate every same-category entity row
    // (multi-KB bodies, overflow pages) AND recompute body_hash64 over each
    // body per write — O(N·body_size) per insert, the measured cause of the
    // quadratic bulk-load curve (#474: 141/s @10K → 7/s @100K). The scan now
    // drives off dedup_signatures alone: small fixed-size rows, band-pruned
    // by the lossless tg_count ratio bound, bodies touched only on a hit.
    // Three pieces make that possible:
    //   1. scope columns, so the scan can filter without joining entities;
    //   2. the (category, workspace_hash, tg_count) band index;
    //   3. a one-time signature backfill for rows the bounded lazy backfill
    //      never reached, making "every ACTIVE row has a signature" an
    //      invariant the scan can rely on. (Writers have maintained
    //      signatures transactionally since v10; archived rows are excluded
    //      — the scan's verify-on-hit self-heals any leftovers.)
    // Backfill cost is one pass over unsigned active rows (~30-60s per
    // 100K on desktop hardware), inside the migration transaction, once.
    ensure_column(
        conn,
        "dedup_signatures",
        "category",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "dedup_signatures",
        "workspace_hash",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    // Scope-sync signature rows that predate the columns (idempotent: rows
    // already synced match the subquery and are rewritten with equal values).
    conn.execute_batch(
        "UPDATE dedup_signatures SET
           category = COALESCE((SELECT e.category FROM entities e
                                WHERE e.id = dedup_signatures.entity_id), category),
           workspace_hash = COALESCE((SELECT e.workspace_hash FROM entities e
                                      WHERE e.id = dedup_signatures.entity_id), workspace_hash);",
    )?;
    // Drop signatures of archived entities: the old scan filtered archived=0
    // via entities; the new signature-driven scan must not see them.
    conn.execute_batch(
        "DELETE FROM dedup_signatures WHERE entity_id IN
           (SELECT id FROM entities WHERE archived = 1);
         DELETE FROM dedup_signature_blobs WHERE entity_id IN
           (SELECT id FROM entities WHERE archived = 1);",
    )?;
    // Backfill missing signatures for every remaining active row.
    {
        let mut missing = conn.prepare(
            "SELECT e.id, e.body_json, e.category, e.workspace_hash
             FROM entities e LEFT JOIN dedup_signatures s ON s.entity_id = e.id
             WHERE e.archived = 0 AND s.entity_id IS NULL",
        )?;
        let rows: Vec<(String, String, String, String)> = missing
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
            .flatten()
            .collect();
        for (id, body, category, ws) in rows {
            let rs = crate::dedup::build_row_signature(&body);
            conn.execute(
                "INSERT OR REPLACE INTO dedup_signature_blobs (entity_id, sig) VALUES (?1, ?2)",
                rusqlite::params![id, rs.sig],
            )?;
            conn.execute(
                "INSERT OR REPLACE INTO dedup_signatures
                 (entity_id, body_len, body_hash, tg_count, histo, category, workspace_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    id,
                    rs.body_len,
                    rs.body_hash,
                    rs.tg_count,
                    rs.histo,
                    category,
                    ws
                ],
            )?;
        }
    }
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_dedup_sig_band
           ON dedup_signatures(category, workspace_hash, tg_count);",
    )?;
    // ── end v17 ──────────────────────────────────────────────────────────

    // ── v18 (#507): dense-search covering index ──────────────────────────
    // See the SCHEMA_VERSION doc comment. Order matters: backfill the
    // signatures FIRST (pure recompute from the stored vector — no model, no
    // key), so the invariant "embedding IS NOT NULL ⟺ emb_sig IS NOT NULL"
    // holds before any query relies on the emb_sig-keyed index.
    {
        let mut missing = conn.prepare(
            "SELECT id, embedding FROM entities \
             WHERE embedding IS NOT NULL AND emb_sig IS NULL",
        )?;
        let rows: Vec<(String, Vec<u8>)> = missing
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .flatten()
            .collect();
        for (id, blob) in rows {
            let vec: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            conn.execute(
                "UPDATE entities SET emb_sig = ?1 WHERE id = ?2",
                rusqlite::params![crate::db::embedding_signature(&vec), id],
            )?;
        }
    }
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_entities_dense_sig
           ON entities(archived, id, emb_sig) WHERE emb_sig IS NOT NULL;",
    )?;
    // ── end v18 ──────────────────────────────────────────────────────────

    // ── v19 (#619 step 2b): int4 refine tier ─────────────────────────────
    // 4-bit scalar-quantized codes (per-row scale + dim/2 nibble bytes) sit
    // between the 1-bit coarse prefilter and the exact-f32 rerank. Same shape
    // as v18: additive column, pure-recompute backfill from the stored
    // vector FIRST (extending the invariant to "embedded ⟺ signed ⟺ coded" —
    // writers set/clear all three together), then a covering index so the
    // per-query code fetch for the coarse pool never touches an entity
    // record (multi-KB body overflow chains).
    ensure_column(conn, "entities", "emb_sig4", "BLOB")?;
    {
        let mut missing = conn.prepare(
            "SELECT id, embedding FROM entities \
             WHERE embedding IS NOT NULL AND emb_sig4 IS NULL",
        )?;
        let rows: Vec<(String, Vec<u8>)> = missing
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .flatten()
            .collect();
        for (id, blob) in rows {
            let vec: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            conn.execute(
                "UPDATE entities SET emb_sig4 = ?1 WHERE id = ?2",
                rusqlite::params![crate::db::embedding_sig4(&vec), id],
            )?;
        }
    }
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_entities_dense_sig4
           ON entities(id, emb_sig4) WHERE emb_sig4 IS NOT NULL;",
    )?;
    // ── end v19 ──────────────────────────────────────────────────────────

    // ── v20 (#682 Temporal RAG): searchable history ──────────────────────
    // Standalone FTS5 over entity_history body terms. Create idempotently only
    // (the base DDL also has this IF NOT EXISTS create for fresh DBs). The
    // representation is selected by the mode-aware write/reindex paths: plaintext
    // for plaintext stores and keyed blind tokens for protected stores. This
    // migration has no key, so it does not backfill history; pre-existing rows
    // are (re)indexed by `reindex_fts` (the `perseus_vault_reindex` tool), which
    // owns the dual encrypted/plaintext path. Fresh installs have empty history,
    // so this is a no-op backfill for them regardless.
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS entity_history_fts USING fts5(body_json);",
    )?;
    // ── end v20 ──────────────────────────────────────────────────────────

    // ── v21 (#683 Keystones): mandatory policy rules table ───────────────
    // New table (the base DDL also has this IF NOT EXISTS create for fresh
    // DBs). No backfill — there is nothing to migrate.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS keystones (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'tenant',
            scope_id TEXT NOT NULL DEFAULT '',
            weight REAL NOT NULL DEFAULT 1.0,
            trust_tier_required INTEGER NOT NULL DEFAULT 2,
            workspace_hash TEXT NOT NULL DEFAULT '',
            author_agent_id TEXT NOT NULL DEFAULT '',
            created_at_unix_ms INTEGER NOT NULL,
            updated_at_unix_ms INTEGER NOT NULL,
            UNIQUE(scope, scope_id, content, workspace_hash)
         );
         CREATE INDEX IF NOT EXISTS idx_keystones_scope
            ON keystones(workspace_hash, scope, scope_id, weight);",
    )?;
    // ── end v21 ──────────────────────────────────────────────────────────

    // ── v22 (#684 Multi-agent scoping): agent registry ───────────────────
    // New table (base DDL has the fresh-DB create). No backfill.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT '',
            trust_tier INTEGER NOT NULL DEFAULT 0,
            fleet_id TEXT NOT NULL DEFAULT '',
            created_at_unix_ms INTEGER NOT NULL,
            updated_at_unix_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_agents_fleet ON agents(fleet_id);",
    )?;
    // ── end v22 ──────────────────────────────────────────────────────────

    // ── v23 (Chancery cross-ref, #6): chancery_writ_id on journal ─────────
    // New column on the journal table. When Chancery wraps an MCP server, it
    // stamps every tools/call payload with `_meta.chancery/lease`; the vault
    // records this in the journal so downstream audit queries can
    // cross-reference against Chancery writ records. ALTER-probe pattern:
    // idempotent and safe on pre-v23 stores. No backfill — legacy rows get
    // '' (the column default) and are populated on all future journal writes.
    ensure_column(conn, "journal", "chancery_writ_id", "TEXT DEFAULT ''")?;
    // ── end v23 ──────────────────────────────────────────────────────────

    // ── v24 (#768 Authorized Action Receipts) ─────────────────────────────
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS authority_manifests (
            id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, workspace_hash TEXT NOT NULL,
            version INTEGER NOT NULL, allowed_capabilities TEXT NOT NULL DEFAULT '[]',
            approval_required_capabilities TEXT NOT NULL DEFAULT '[]', scope_anchors TEXT NOT NULL DEFAULT '[]',
            approver_principals TEXT NOT NULL DEFAULT '[]', allowed_inbound_principals TEXT NOT NULL DEFAULT '[]',
            permitted_external_ref_prefixes TEXT NOT NULL DEFAULT '[]', max_parallel_actions INTEGER NOT NULL DEFAULT 1,
            mode TEXT NOT NULL DEFAULT 'shadow', expires_at_unix_ms INTEGER, revoked_at_unix_ms INTEGER,
            created_at_unix_ms INTEGER NOT NULL,
            UNIQUE(agent_id, workspace_hash, version));
         CREATE INDEX IF NOT EXISTS idx_authority_active ON authority_manifests(agent_id, workspace_hash, revoked_at_unix_ms, version DESC);
         CREATE TABLE IF NOT EXISTS authorized_actions (
            id TEXT PRIMARY KEY, manifest_id TEXT NOT NULL REFERENCES authority_manifests(id), manifest_version INTEGER NOT NULL DEFAULT 0,
            agent_id TEXT NOT NULL, workspace_hash TEXT NOT NULL, scope_anchor TEXT NOT NULL, external_ref TEXT NOT NULL DEFAULT '',
            capability TEXT NOT NULL, action_key TEXT NOT NULL, intent_hash TEXT NOT NULL, outcome_hash TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL, approval_required INTEGER NOT NULL DEFAULT 0, approval_ref TEXT NOT NULL DEFAULT '',
            created_at_unix_ms INTEGER NOT NULL, updated_at_unix_ms INTEGER NOT NULL);
         CREATE INDEX IF NOT EXISTS idx_authorized_actions_scope ON authorized_actions(workspace_hash, action_key, status);
         CREATE TABLE IF NOT EXISTS authorized_action_leases (
            id TEXT PRIMARY KEY, action_id TEXT NOT NULL REFERENCES authorized_actions(id), workspace_hash TEXT NOT NULL,
            action_key TEXT NOT NULL, holder_id TEXT NOT NULL, expires_at_unix_ms INTEGER NOT NULL,
            released_at_unix_ms INTEGER, created_at_unix_ms INTEGER NOT NULL);
         CREATE INDEX IF NOT EXISTS idx_authorized_action_leases_active ON authorized_action_leases(workspace_hash, action_key, released_at_unix_ms, expires_at_unix_ms);",
    )?;
    ensure_column(
        conn,
        "authority_manifests",
        "allowed_inbound_principals",
        "TEXT NOT NULL DEFAULT '[]'",
    )?;
    ensure_column(
        conn,
        "authority_manifests",
        "permitted_external_ref_prefixes",
        "TEXT NOT NULL DEFAULT '[]'",
    )?;
    ensure_column(
        conn,
        "authority_manifests",
        "max_parallel_actions",
        "INTEGER NOT NULL DEFAULT 1",
    )?;
    ensure_column(
        conn,
        "authorized_actions",
        "manifest_version",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    ensure_column(
        conn,
        "authorized_actions",
        "external_ref",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "authorized_actions",
        "outcome_hash",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "authority_manifests",
        "capability_constraints_json",
        "TEXT NOT NULL DEFAULT '{}'",
    )?;
    ensure_column(
        conn,
        "authorized_actions",
        "resource_constraints_json",
        "TEXT NOT NULL DEFAULT '{}'",
    )?;
    ensure_column(
        conn,
        "authorized_actions",
        "resource_constraints_hash",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    // #10 resource constraints are hash-only lifecycle metadata. Legacy rows
    // remain readable with NULL/empty defaults; opted-in actions populate them.
    // ── end v24 ──────────────────────────────────────────────────────────

    // ── v25 (#811 Immutable artifacts) ───────────────────────────────────
    // Digest semantics (#835): sha256 is byte identity only — it proves the
    // bytes are the bytes; validity/authority/freshness come from binding state.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS artifacts (
            sha256 TEXT PRIMARY KEY,
            content_b64 TEXT NOT NULL,
            byte_length INTEGER NOT NULL,
            created_at_unix_ms INTEGER NOT NULL
         );
         CREATE TABLE IF NOT EXISTS artifact_bindings (
            binding_id TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL REFERENCES artifacts(sha256),
            mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
            workspace_hash TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT '',
            visibility TEXT NOT NULL DEFAULT 'workspace',
            origin_json TEXT NOT NULL DEFAULT '{}',
            external_refs_json TEXT NOT NULL DEFAULT '[]',
            retention_policy TEXT NOT NULL DEFAULT '',
            representation_kind TEXT NOT NULL DEFAULT 'original',
            derived_from_sha256 TEXT NOT NULL DEFAULT '',
            derivation_kind TEXT NOT NULL DEFAULT '',
            derivation_version TEXT NOT NULL DEFAULT '',
            created_at_unix_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_artifact_bindings_scope
           ON artifact_bindings(sha256, workspace_hash, visibility, created_at_unix_ms DESC);
         CREATE INDEX IF NOT EXISTS idx_artifact_bindings_derived_from
           ON artifact_bindings(derived_from_sha256);",
    )?;
    // ── end v25 ──────────────────────────────────────────────────────────

    // ── v26 (#849 rejected-value tombstones) ─────────────────────────────
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS rejected_value_tombstones (
            id TEXT PRIMARY KEY,
            workspace_hash TEXT NOT NULL DEFAULT '',
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            value_sha256 TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            evidence_ref TEXT NOT NULL DEFAULT '',
            author_agent_id TEXT NOT NULL DEFAULT '',
            created_at_unix_ms INTEGER NOT NULL,
            expires_at_unix_ms INTEGER
         );
         CREATE UNIQUE INDEX IF NOT EXISTS idx_rejected_tombstones_identity
           ON rejected_value_tombstones(workspace_hash, subject, predicate, value_sha256);
         CREATE INDEX IF NOT EXISTS idx_rejected_tombstones_scope
           ON rejected_value_tombstones(workspace_hash, expires_at_unix_ms);",
    )?;
    // ── end v26 ──────────────────────────────────────────────────────────

    // ── v27 (#880 epistemic states) ───────────────────────────────────────
    // Trust axis on entities, orthogonal to lifecycle `status`. Default
    // 'candidate' is the fail-closed reading for legacy rows lacking
    // admission evidence; writers set verified/corroborated explicitly.
    ensure_column(
        conn,
        "entities",
        "epistemic_state",
        "TEXT DEFAULT 'candidate'",
    )?;
    // ── end v27 ──────────────────────────────────────────────────────────

    // ── v31 (#872 retrieval concentration / contamination telemetry) ─────
    // Read-only observability substrate: served events (one row per
    // delivered entity per recall), per-arm audit rows (candidate /
    // re-entry / delivered counts per recall mode), and displacement
    // events (cooldown/diversity controls removing entities). All three
    // are write-on-read side effects of serving (like retrieval_count),
    // bounded by retention pruning in the recording helpers.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS served_events (
            id TEXT PRIMARY KEY,
            ts_unix_ms INTEGER NOT NULL,
            batch_id TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT '',
            workspace_hash TEXT NOT NULL DEFAULT '',
            entity_id TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT '',
            key TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            verified INTEGER NOT NULL DEFAULT 0,
            certainty REAL NOT NULL DEFAULT 0.5,
            mode TEXT NOT NULL DEFAULT '',
            query TEXT NOT NULL DEFAULT '',
            query_class TEXT NOT NULL DEFAULT '',
            tokens_est INTEGER NOT NULL DEFAULT 0,
            slot INTEGER NOT NULL DEFAULT 0
         );
         CREATE INDEX IF NOT EXISTS idx_served_events_ts ON served_events(ts_unix_ms);
         CREATE INDEX IF NOT EXISTS idx_served_events_entity ON served_events(entity_id);
         CREATE TABLE IF NOT EXISTS recall_arm_audits (
            id TEXT PRIMARY KEY,
            ts_unix_ms INTEGER NOT NULL,
            mode TEXT NOT NULL,
            arm TEXT NOT NULL,
            candidates INTEGER NOT NULL,
            reentry_candidates INTEGER NOT NULL,
            delivered INTEGER NOT NULL,
            profile TEXT NOT NULL DEFAULT '',
            workspace_hash TEXT NOT NULL DEFAULT '',
            query_hash TEXT NOT NULL DEFAULT ''
         );
         CREATE INDEX IF NOT EXISTS idx_arm_audits_ts ON recall_arm_audits(ts_unix_ms);
         CREATE TABLE IF NOT EXISTS displacement_events (
            id TEXT PRIMARY KEY,
            ts_unix_ms INTEGER NOT NULL,
            entity_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            was_sole_evidence INTEGER NOT NULL DEFAULT 0,
            mode TEXT NOT NULL DEFAULT '',
            workspace_hash TEXT NOT NULL DEFAULT '',
            query TEXT NOT NULL DEFAULT ''
         );
         CREATE INDEX IF NOT EXISTS idx_displacement_ts ON displacement_events(ts_unix_ms);",
    )?;
    // ── end v31 ──────────────────────────────────────────────────────────

    // ── v32 (#889 keystone-suggestion queue) ─────────────────────────────
    // Candidate directive/keystone suggestions extracted from `correct`
    // captures by word-boundary-anchored patterns. Suggestions are never
    // policy: only an explicit operator decision (`approve`) promotes one to
    // the keystones table, preserving the governance gate.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS keystone_suggestions (
            id TEXT PRIMARY KEY,
            source_entity_id TEXT NOT NULL,
            source_category TEXT NOT NULL DEFAULT 'correction',
            instruction TEXT NOT NULL,
            pattern_locale TEXT NOT NULL,
            matched_pattern TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at_unix_ms INTEGER NOT NULL,
            decided_at_unix_ms INTEGER,
            decided_by TEXT,
            workspace_hash TEXT NOT NULL DEFAULT '',
            UNIQUE(source_entity_id, instruction)
         );
         CREATE INDEX IF NOT EXISTS idx_ksug_status_created
            ON keystone_suggestions(status, created_at_unix_ms);
         CREATE INDEX IF NOT EXISTS idx_ksug_source
            ON keystone_suggestions(source_entity_id);",
    )?;
    // ── end v32 ──────────────────────────────────────────────────────────

    // ── v33 (#885): optional quantized embedding storage ─────────────────
    // New tables (the base DDL also has these IF NOT EXISTS creates for
    // fresh DBs). No backfill: the `embedding_format` record is written by
    // the reindex path (perseus_vault_embed quant_mode) and by open() for
    // fresh stores declared quantized from the start; the snapshot is
    // created only by the quantization step itself.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS embedding_format (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            format TEXT NOT NULL,
            updated_at_unix_ms INTEGER NOT NULL
         );
         CREATE TABLE IF NOT EXISTS entities_embedding_snapshot (
            id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            created_at_unix_ms INTEGER NOT NULL
         );",
    )?;
    // ── end v33 ──────────────────────────────────────────────────────────

    // ── v34 (#874): reviewable write-quarantine hold ─────────────────────
    // New table only (base DDL also creates it for fresh DBs); no backfill —
    // nothing existed before v34 to migrate.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS write_quarantine (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            body_json TEXT NOT NULL,
            links TEXT NOT NULL DEFAULT '[]',
            tags TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'active',
            entity_type TEXT NOT NULL DEFAULT 'insight',
            importance REAL NOT NULL DEFAULT 0.5,
            certainty REAL NOT NULL DEFAULT 0.5,
            visibility TEXT NOT NULL DEFAULT 'workspace',
            topic_path TEXT NOT NULL DEFAULT '',
            always_on INTEGER NOT NULL DEFAULT 0,
            agent_id TEXT NOT NULL DEFAULT '',
            workspace_hash TEXT NOT NULL DEFAULT '',
            valid_from_unix_ms INTEGER,
            valid_to_unix_ms INTEGER,
            interference_score REAL NOT NULL,
            interference_json TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT 'interference',
            created_at_unix_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_write_quarantine_ws
          ON write_quarantine(workspace_hash, created_at_unix_ms);",
    )?;
    // ── end v34 ──────────────────────────────────────────────────────────

    // ── v35 (#871): durable long-running operation states ────────────────
    // New tables only (base DDL also creates them for fresh DBs); no
    // backfill — nothing existed before v35 to migrate.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS op_runs (
            id TEXT PRIMARY KEY,
            op_type TEXT NOT NULL,
            state TEXT NOT NULL,
            partial INTEGER NOT NULL DEFAULT 0,
            timeout INTEGER NOT NULL DEFAULT 0,
            stale INTEGER NOT NULL DEFAULT 0,
            scope TEXT NOT NULL DEFAULT '',
            input_digest TEXT NOT NULL DEFAULT '',
            items_total INTEGER NOT NULL DEFAULT 0,
            items_done INTEGER NOT NULL DEFAULT 0,
            items_failed INTEGER NOT NULL DEFAULT 0,
            items_unattempted INTEGER NOT NULL DEFAULT 0,
            error_class TEXT NOT NULL DEFAULT '',
            error_detail TEXT NOT NULL DEFAULT '',
            receipt TEXT NOT NULL DEFAULT '',
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 2,
            parent_run_id TEXT NOT NULL DEFAULT '',
            created_by TEXT NOT NULL DEFAULT '',
            created_at_unix_ms INTEGER NOT NULL,
            started_at_unix_ms INTEGER,
            updated_at_unix_ms INTEGER NOT NULL,
            finished_at_unix_ms INTEGER
         );
         CREATE INDEX IF NOT EXISTS idx_op_runs_state
          ON op_runs(state, updated_at_unix_ms);
         CREATE INDEX IF NOT EXISTS idx_op_runs_type
          ON op_runs(op_type, created_at_unix_ms);
         CREATE TABLE IF NOT EXISTS op_run_items (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            item_ref TEXT NOT NULL,
            item_digest TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL,
            receipt_ref TEXT NOT NULL DEFAULT '',
            error_class TEXT NOT NULL DEFAULT '',
            error_detail TEXT NOT NULL DEFAULT '',
            retry_count INTEGER NOT NULL DEFAULT 0,
            created_at_unix_ms INTEGER NOT NULL,
            updated_at_unix_ms INTEGER NOT NULL,
            finished_at_unix_ms INTEGER,
            UNIQUE(run_id, item_ref)
         );
         CREATE INDEX IF NOT EXISTS idx_op_run_items_run
          ON op_run_items(run_id, state);",
    )?;
    // ── end v35 ──────────────────────────────────────────────────────────

    // ── v36 (#875): learned anticipation — preload usage telemetry ────────
    // New tables only; no backfill (no preload events existed before v36).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS preload_events (
            id TEXT PRIMARY KEY,
            ts INTEGER NOT NULL,
            context_hash TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT '',
            entity_id TEXT NOT NULL,
            trigger_ref TEXT NOT NULL,
            workspace_hash TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL DEFAULT '',
            la_before INTEGER NOT NULL,
            used INTEGER,
            resolved_ts INTEGER
         );
         CREATE INDEX IF NOT EXISTS idx_preload_events_resolve
          ON preload_events(used, ts);
         CREATE INDEX IF NOT EXISTS idx_preload_events_entity
          ON preload_events(entity_id, ts);
         CREATE TABLE IF NOT EXISTS preload_sessions (
            session_id TEXT PRIMARY KEY,
            anchor_ts INTEGER NOT NULL,
            preloaded_n INTEGER NOT NULL,
            used_n INTEGER NOT NULL,
            missed_n INTEGER NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            miss_rate REAL NOT NULL,
            context_words TEXT NOT NULL DEFAULT '[]',
            missed_by_trigger_json TEXT NOT NULL DEFAULT '{}',
            missed_ids_json TEXT NOT NULL DEFAULT '[]',
            resolved_ts INTEGER NOT NULL
         );
         CREATE TABLE IF NOT EXISTS preload_proposals (
            id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            trigger_ref TEXT NOT NULL,
            suggestion TEXT NOT NULL,
            rationale TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL,
            created_ts INTEGER NOT NULL,
            decided_ts INTEGER,
            decided_by TEXT NOT NULL DEFAULT '',
            applied_ts INTEGER,
            journal_event_id TEXT NOT NULL DEFAULT ''
         );
         CREATE INDEX IF NOT EXISTS idx_preload_proposals_state
          ON preload_proposals(state, created_ts);",
    )?;
    // ── end v36 ──────────────────────────────────────────────────────────

    // ── v37 (#919): prospective query hints ────────────────────────────
    // Advisory retrieval metadata column on entities (JSON array of hint
    // strings, default no hints). New column only; every existing row keeps
    // an empty hints array. FTS5 indexing appends hints to the indexed text
    // when present (db.rs::fts_indexed_text) — the column itself is opaque
    // to recall until then. Idempotent via ensure_column.
    ensure_column(conn, "entities", "hints", "TEXT NOT NULL DEFAULT '[]'")?;
    // ── end v37 ────────────────────────────────────────────────────────

    // ── v38 (#930): scheduled recall evaluation ────────────────────────
    // Durable eval history: bounded metric snapshots of quality runs
    // (nightly curation + midday eval), regression breach records, and the
    // nightly after-action summary. Booleans/counters/digests/rates only —
    // raw prompts, bodies, tool arguments, and credentials never land here
    // (the harness report already excludes them; record validates).
    // Also under v38 (#940): court-of-record rulings (same migration level;
    // both tables are probed idempotently below).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS eval_runs (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL DEFAULT '',
            eval_kind TEXT NOT NULL,
            suite TEXT NOT NULL DEFAULT 'memory-quality-v1',
            status TEXT NOT NULL,
            run_at_unix_ms INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            manifest_digest TEXT NOT NULL DEFAULT '',
            binary_digest TEXT NOT NULL DEFAULT '',
            harness_version TEXT NOT NULL DEFAULT '',
            checks_passed INTEGER NOT NULL DEFAULT 0,
            checks_total INTEGER NOT NULL DEFAULT 0,
            accuracy REAL NOT NULL DEFAULT 0,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            maintain_summary_json TEXT NOT NULL DEFAULT '',
            breaches_json TEXT NOT NULL DEFAULT '[]',
            regressed INTEGER NOT NULL DEFAULT 0,
            created_by TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_eval_runs_kind
            ON eval_runs(eval_kind, run_at_unix_ms);
        CREATE INDEX IF NOT EXISTS idx_eval_runs_regressed
            ON eval_runs(regressed, run_at_unix_ms);
        CREATE TABLE IF NOT EXISTS court_rulings (
            id TEXT PRIMARY KEY,
            pair_fingerprint TEXT NOT NULL,
            winner_id TEXT NOT NULL,
            loser_id TEXT NOT NULL,
            ruling TEXT NOT NULL,
            rationale TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            decided_by TEXT NOT NULL DEFAULT '',
            supersede_receipt TEXT NOT NULL DEFAULT '',
            created_at_unix_ms INTEGER NOT NULL,
            reversed_at_unix_ms INTEGER,
            reversed_by TEXT NOT NULL DEFAULT '',
            UNIQUE(pair_fingerprint, status)
        );
        CREATE INDEX IF NOT EXISTS idx_court_rulings_pair
            ON court_rulings(pair_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_court_rulings_status
            ON court_rulings(status, created_at_unix_ms);",
    )?;
    // ── v39 (#1050): provenance-admission containment replay ─────────────
    // Exact-content admission ledger for connector re-ingest. A row records
    // that a canonical (category, key, body) was already admitted into a live
    // entity, so re-ingesting an unchanged feed becomes zero-work
    // revalidation instead of extraction/admission churn. The gate is a
    // containment shortcut only — admitted documents still flow through the
    // normal remember/admission path, and the row is refreshed only on a
    // successful admission.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS admission_replay (
            fingerprint TEXT NOT NULL,
            source TEXT NOT NULL,
            first_seen_ms INTEGER NOT NULL,
            last_seen_ms INTEGER NOT NULL,
            covered_entity_id TEXT NOT NULL,
            admitted INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (fingerprint, source)
         );
         CREATE INDEX IF NOT EXISTS idx_admission_replay_source
            ON admission_replay(source, last_seen_ms);
         CREATE INDEX IF NOT EXISTS idx_admission_replay_entity
            ON admission_replay(covered_entity_id);",
    )?;
    // ── end v39 ────────────────────────────────────────────────────────

    // ── v40 (#1048): extraction-loss net ─────────────────────────────────
    // Residual spans (sentences an extractor missed, retained verbatim with
    // provenance), provisional query keys (confirmed retries serve
    // first-pass), and lossy-unit marks (repeated under-coverage, repaired
    // append-only on touch).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS residual_spans (
            id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            span_text TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            max_coverage REAL NOT NULL DEFAULT 0,
            coverage_mode TEXT NOT NULL DEFAULT 'token',
            status TEXT NOT NULL DEFAULT 'active',
            lossy_count INTEGER NOT NULL DEFAULT 0,
            created_ms INTEGER NOT NULL,
            last_served_ms INTEGER
         );
         CREATE INDEX IF NOT EXISTS idx_residual_spans_entity
            ON residual_spans(entity_id, status);
         CREATE TABLE IF NOT EXISTS query_keys (
            fingerprint TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            entity_ids TEXT NOT NULL,
            confirmed_ms INTEGER NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 0
         );
         CREATE TABLE IF NOT EXISTS lossy_units (
            entity_id TEXT PRIMARY KEY,
            lossy_count INTEGER NOT NULL DEFAULT 0,
            marked_at_ms INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'lossy'
         );",
    )?;
    // ── end v38 ────────────────────────────────────────────────────────

    // ── v51 (#1060 seal-style tamper evidence) ───────────────────────────
    // Seals: SHA-256 commitments over content recorded at write/export time,
    // stored OUTSIDE the sealed content itself (hash + label only — no content
    // leak). Compare-on-recall/reload surfaces any mismatch as a tamper event.
    // Integrity ≠ truth by design: a seal proves unchanged-since-sealed, never
    // true-when-written. See docs/seals-tamper-evidence.md.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memory_seals (
            seal_id TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            label TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL DEFAULT 'entity',
            sha256 TEXT NOT NULL,
            workspace_hash TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT '',
            created_at_unix_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_memory_seals_target
            ON memory_seals(target_id, workspace_hash, created_at_unix_ms);",
    )?;
    // ── end v51 ──────────────────────────────────────────────────────────

    // ── v52 (#1064 typed provenance edges) ───────────────────────────────
    // Parameter-level lineage for high-risk tool arguments: where did this
    // value come from (Agent-Sentry pattern, arXiv:2603.22868). One row per
    // (entity, parameter path) assertion; source_ref may cite the producing
    // entity so forged lineage is detectable at query time.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS param_lineage (
            lineage_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            param_path TEXT NOT NULL,
            source_kind TEXT NOT NULL DEFAULT 'manual',
            source_ref TEXT NOT NULL DEFAULT '',
            workspace_hash TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT '',
            asserted_at_unix_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_param_lineage_entity
            ON param_lineage(entity_id, param_path);",
    )?;
    // ── end v52 ──────────────────────────────────────────────────────────

    // ── v53 (#1065 intent-aware typed-relational traversal) ──────────────
    // Ablation substrate: one row per typed-traversal run so each relation
    // view (temporal/causal/entity/semantic) can report whether it earns its
    // token cost. Hash-only query identity; bounded by retention pruning.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS traversal_runs (
            run_id TEXT PRIMARY KEY,
            query_hash TEXT NOT NULL,
            intent TEXT NOT NULL DEFAULT '',
            view TEXT NOT NULL DEFAULT '',
            selected_count INTEGER NOT NULL DEFAULT 0,
            rejected_count INTEGER NOT NULL DEFAULT 0,
            tokens_selected INTEGER NOT NULL DEFAULT 0,
            tokens_rejected INTEGER NOT NULL DEFAULT 0,
            created_at_unix_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_traversal_runs_view
            ON traversal_runs(view, created_at_unix_ms);",
    )?;
    // ── end v53 ──────────────────────────────────────────────────────────

    // ── v54 (#1066 model-upgrade inheritance receipts) ──────────────────
    // Identity/vessel split (arXiv:2603.04740): a subject identity survives
    // model changes; incarnations record which model instance served it and
    // when. Inheritance receipts make the boundary between incarnations an
    // explicit, governed, auditable transition.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS subject_identities (
            subject_id TEXT PRIMARY KEY,
            label TEXT NOT NULL DEFAULT '',
            created_at_unix_ms INTEGER NOT NULL
         );
         CREATE TABLE IF NOT EXISTS model_incarnations (
            incarnation_id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT '',
            started_at_unix_ms INTEGER NOT NULL,
            ended_at_unix_ms INTEGER,
            departure_reason TEXT NOT NULL DEFAULT '',
            UNIQUE(subject_id, model_id, started_at_unix_ms)
         );
         CREATE INDEX IF NOT EXISTS idx_incarnations_subject
            ON model_incarnations(subject_id, started_at_unix_ms);",
    )?;
    // ── end v54 ──────────────────────────────────────────────────────────

    // ── v55 (#1080 signed transitions + poison labels) ─────────────────
    // MutMem (arXiv:2608.02843): retrieval-relevant mutations commit as
    // Ed25519-signed transitions in a no-fork chain; poison-likely content is
    // retained with signed, revisable labels consumed by recall as trust
    // evidence. `seed_b64` holds the epoch's Ed25519 seed so the writer can
    // sign in-process; at-rest posture matches the AES key file (same
    // operator trust domain, documented in docs/specs/signed-transitions.md).
    // The UNIQUE predecessor index makes forks unrepresentable at the storage
    // level: at most one successor may claim any predecessor, and at most one
    // record may be the genesis (empty predecessor).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS signer_epochs (
            epoch INTEGER PRIMARY KEY,
            public_key_b64 TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            seed_b64 TEXT NOT NULL,
            created_at_unix_ms INTEGER NOT NULL
         );
         CREATE TABLE IF NOT EXISTS signed_transitions (
            id TEXT PRIMARY KEY,
            terminal_node TEXT NOT NULL,
            signer_epoch INTEGER NOT NULL,
            signer_fingerprint TEXT NOT NULL,
            old_value_json TEXT NOT NULL,
            new_value_json TEXT NOT NULL,
            commitment_old TEXT NOT NULL,
            commitment_new TEXT NOT NULL,
            predecessor_hash TEXT NOT NULL DEFAULT '',
            signature_b64 TEXT NOT NULL,
            chain_hash TEXT NOT NULL,
            created_at_unix_ms INTEGER NOT NULL
         );
         CREATE UNIQUE INDEX IF NOT EXISTS idx_signed_transitions_predecessor
            ON signed_transitions(predecessor_hash);
         CREATE INDEX IF NOT EXISTS idx_signed_transitions_epoch
            ON signed_transitions(signer_epoch, created_at_unix_ms);
         CREATE TABLE IF NOT EXISTS poison_labels (
            entity_id TEXT PRIMARY KEY,
            level TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            transition_id TEXT NOT NULL DEFAULT '',
            updated_at_unix_ms INTEGER NOT NULL
         );",
    )?;
    // ── end v55 ──────────────────────────────────────────────────────────

    // ── v56 (#1134 task/action lineage) ──────────────────────────────────
    // Optional AAR sidecar. The current row is mutable only through an
    // exact-head CAS; transitions are append-only history and contain only
    // bounded/hash-bound admission state.
    ensure_column(
        conn,
        "authorized_actions",
        "lineage_id",
        "TEXT NOT NULL DEFAULT \"\"",
    )?;
    ensure_column(
        conn,
        "authorized_actions",
        "lineage_transition_id",
        "TEXT NOT NULL DEFAULT \"\"",
    )?;
    ensure_column(
        conn,
        "authorized_actions",
        "lineage_outcome",
        "TEXT NOT NULL DEFAULT \"\"",
    )?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS action_lineages (
            lineage_id TEXT PRIMARY KEY,
            parent_lineage_id TEXT NOT NULL DEFAULT \"\",
            parent_head_digest TEXT NOT NULL DEFAULT \"\",
            workspace_hash TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            authority_manifest_id TEXT NOT NULL,
            authority_manifest_version INTEGER NOT NULL,
            policy_version TEXT NOT NULL,
            continuation_state_json TEXT NOT NULL,
            continuation_state_digest TEXT NOT NULL,
            head_digest TEXT NOT NULL,
            budget_limit INTEGER NOT NULL,
            impact_limit INTEGER NOT NULL,
            budget_spent INTEGER NOT NULL,
            impact_units INTEGER NOT NULL,
            expires_at_unix_ms INTEGER,
            revoked_at_unix_ms INTEGER,
            created_at_unix_ms INTEGER NOT NULL,
            updated_at_unix_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_action_lineages_scope
            ON action_lineages(workspace_hash, agent_id, updated_at_unix_ms);
         CREATE TABLE IF NOT EXISTS action_lineage_transitions (
            transition_id TEXT PRIMARY KEY,
            lineage_id TEXT NOT NULL REFERENCES action_lineages(lineage_id),
            parent_lineage_id TEXT NOT NULL DEFAULT \"\",
            action_id TEXT NOT NULL REFERENCES authorized_actions(id),
            idempotency_key_digest TEXT NOT NULL,
            request_digest TEXT NOT NULL,
            parent_head_digest TEXT NOT NULL,
            head_digest TEXT NOT NULL,
            continuation_state_json TEXT NOT NULL,
            continuation_state_digest TEXT NOT NULL,
            workspace_hash TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            authority_manifest_id TEXT NOT NULL,
            authority_manifest_version INTEGER NOT NULL,
            policy_version TEXT NOT NULL,
            outcome TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            budget_cost INTEGER NOT NULL,
            impact_units INTEGER NOT NULL,
            budget_limit INTEGER NOT NULL,
            impact_limit INTEGER NOT NULL,
            budget_spent INTEGER NOT NULL,
            impact_spent INTEGER NOT NULL DEFAULT 0,
            expires_at_unix_ms INTEGER,
            revoked_at_unix_ms INTEGER,
            created_at_unix_ms INTEGER NOT NULL,
            UNIQUE(lineage_id, idempotency_key_digest)
         );
         CREATE INDEX IF NOT EXISTS idx_action_lineage_transitions_lineage
            ON action_lineage_transitions(lineage_id, created_at_unix_ms);
         CREATE TRIGGER IF NOT EXISTS action_lineage_transitions_no_update
            BEFORE UPDATE ON action_lineage_transitions
            BEGIN SELECT RAISE(ABORT, \"action lineage history is append-only\"); END;
         CREATE TRIGGER IF NOT EXISTS action_lineage_transitions_no_delete
            BEFORE DELETE ON action_lineage_transitions
            BEGIN SELECT RAISE(ABORT, \"action lineage history is append-only\"); END;",
    )?;

    ensure_column(
        conn,
        "action_lineage_transitions",
        "budget_limit",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    ensure_column(
        conn,
        "action_lineage_transitions",
        "impact_limit",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    ensure_column(
        conn,
        "action_lineage_transitions",
        "impact_spent",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    // v57 (#1141 provider-native source identity and event lifecycle). Provider
    // bodies never enter these tables: only bounded identity, lineage, timing,
    // scope, digest, and hash-only receipt fields are retained.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS provider_sources (
            source_id TEXT PRIMARY KEY,
            workspace_hash TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL,
            kind TEXT NOT NULL,
            external_id TEXT NOT NULL,
            canonical_uri TEXT,
            thread_id TEXT,
            parent_id TEXT,
            provider_event_id TEXT,
            author TEXT,
            revision TEXT NOT NULL,
            observed_at_unix_ms INTEGER,
            provider_created_at_unix_ms INTEGER,
            provider_updated_at_unix_ms INTEGER,
            content_sha256 TEXT,
            source_span_ref TEXT,
            visibility TEXT NOT NULL DEFAULT 'workspace',
            retention_policy TEXT,
            capture_method TEXT NOT NULL DEFAULT 'event_feed',
            authority_agent_id TEXT NOT NULL DEFAULT '',
            entity_id TEXT,
            state TEXT NOT NULL DEFAULT 'active',
            deleted_at_unix_ms INTEGER,
            current_event_id TEXT NOT NULL,
            receipt_digest TEXT NOT NULL,
            recorded_at_unix_ms INTEGER NOT NULL,
            updated_at_unix_ms INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_provider_sources_identity
            ON provider_sources(workspace_hash, provider, external_id);
        CREATE INDEX IF NOT EXISTS idx_provider_sources_entity
            ON provider_sources(workspace_hash, entity_id);
        CREATE TABLE IF NOT EXISTS provider_source_events (
            event_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES provider_sources(source_id),
            workspace_hash TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL,
            kind TEXT NOT NULL,
            external_id TEXT NOT NULL,
            canonical_uri TEXT,
            thread_id TEXT,
            parent_id TEXT,
            provider_event_id TEXT,
            author TEXT,
            revision TEXT NOT NULL,
            expected_revision TEXT,
            event_type TEXT NOT NULL,
            observed_at_unix_ms INTEGER,
            provider_created_at_unix_ms INTEGER,
            provider_updated_at_unix_ms INTEGER,
            content_sha256 TEXT,
            source_span_ref TEXT,
            visibility TEXT NOT NULL DEFAULT 'workspace',
            retention_policy TEXT,
            capture_method TEXT NOT NULL DEFAULT 'event_feed',
            authority_agent_id TEXT NOT NULL DEFAULT '',
            entity_id TEXT,
            state_after TEXT NOT NULL,
            deleted_at_unix_ms INTEGER,
            previous_revision TEXT,
            request_digest TEXT NOT NULL,
            receipt_digest TEXT NOT NULL,
            recorded_at_unix_ms INTEGER NOT NULL,
            UNIQUE(source_id, revision)
        );
        CREATE INDEX IF NOT EXISTS idx_provider_source_events_scope
            ON provider_source_events(workspace_hash, provider, external_id, recorded_at_unix_ms);
        CREATE INDEX IF NOT EXISTS idx_provider_source_events_source
            ON provider_source_events(source_id, recorded_at_unix_ms);
        CREATE TRIGGER IF NOT EXISTS provider_source_events_no_update
            BEFORE UPDATE ON provider_source_events
            BEGIN SELECT RAISE(ABORT, 'provider source event history is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS provider_source_events_no_delete
            BEFORE DELETE ON provider_source_events
            BEGIN SELECT RAISE(ABORT, 'provider source event history is append-only'); END;",
    )?;

    // v58 (#1142 declared graph ingestion). Manifests and edge rows keep
    // source revision, digest, scope, validity, origin, support state, and
    // replacement or tombstone history separate from legacy entities.links.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS declared_graph_nodes (
            node_id TEXT PRIMARY KEY,
            workspace_hash TEXT NOT NULL,
            namespace TEXT NOT NULL,
            canonical_id TEXT NOT NULL,
            node_type TEXT NOT NULL,
            external_ref TEXT,
            state TEXT NOT NULL,
            created_at_unix_ms INTEGER NOT NULL,
            updated_at_unix_ms INTEGER NOT NULL,
            UNIQUE(workspace_hash, namespace, canonical_id)
        );
        CREATE INDEX IF NOT EXISTS idx_declared_graph_nodes_scope
            ON declared_graph_nodes(workspace_hash, namespace, canonical_id);
        CREATE TABLE IF NOT EXISTS declared_graph_manifests (
            manifest_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            workspace_hash TEXT NOT NULL,
            source_key TEXT NOT NULL,
            revision TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            source_span_ref TEXT,
            policy TEXT NOT NULL,
            origin TEXT NOT NULL,
            state TEXT NOT NULL,
            request_digest TEXT NOT NULL,
            node_ids_json TEXT NOT NULL,
            valid_from_unix_ms INTEGER,
            valid_to_unix_ms INTEGER,
            recorded_at_unix_ms INTEGER NOT NULL,
            updated_at_unix_ms INTEGER NOT NULL,
            tombstoned_at_unix_ms INTEGER,
            UNIQUE(workspace_hash, source_key, revision)
        );
        CREATE INDEX IF NOT EXISTS idx_declared_graph_manifests_current
            ON declared_graph_manifests(workspace_hash, source_key, state);
        CREATE TABLE IF NOT EXISTS declared_graph_edges (
            edge_id TEXT PRIMARY KEY,
            manifest_id TEXT NOT NULL REFERENCES declared_graph_manifests(manifest_id),
            workspace_hash TEXT NOT NULL,
            from_node_id TEXT NOT NULL REFERENCES declared_graph_nodes(node_id),
            to_node_id TEXT NOT NULL REFERENCES declared_graph_nodes(node_id),
            predicate TEXT NOT NULL,
            direction TEXT NOT NULL,
            context TEXT,
            source_span_ref TEXT,
            origin TEXT NOT NULL,
            support_state TEXT NOT NULL,
            attested_by TEXT,
            attestation_ref TEXT,
            valid_from_unix_ms INTEGER,
            valid_to_unix_ms INTEGER,
            state TEXT NOT NULL,
            recorded_at_unix_ms INTEGER NOT NULL,
            updated_at_unix_ms INTEGER NOT NULL,
            UNIQUE(manifest_id, from_node_id, to_node_id, predicate, direction)
        );
        CREATE INDEX IF NOT EXISTS idx_declared_graph_edges_scope
            ON declared_graph_edges(workspace_hash, state, recorded_at_unix_ms);
        CREATE INDEX IF NOT EXISTS idx_declared_graph_edges_manifest
            ON declared_graph_edges(manifest_id, state);
        CREATE TABLE IF NOT EXISTS declared_graph_manifest_events (
            event_id TEXT PRIMARY KEY,
            manifest_id TEXT NOT NULL REFERENCES declared_graph_manifests(manifest_id),
            source_id TEXT NOT NULL,
            workspace_hash TEXT NOT NULL,
            source_key TEXT NOT NULL,
            revision TEXT NOT NULL,
            operation TEXT NOT NULL,
            state_after TEXT NOT NULL,
            request_digest TEXT NOT NULL,
            actor_agent_id TEXT NOT NULL,
            receipt_digest TEXT NOT NULL,
            recorded_at_unix_ms INTEGER NOT NULL,
            UNIQUE(manifest_id, request_digest, operation)
        );
        CREATE INDEX IF NOT EXISTS idx_declared_graph_events_source
            ON declared_graph_manifest_events(workspace_hash, source_key, recorded_at_unix_ms);
        CREATE TRIGGER IF NOT EXISTS declared_graph_manifest_events_no_update
            BEFORE UPDATE ON declared_graph_manifest_events
            BEGIN SELECT RAISE(ABORT, \"declared graph manifest history is append only\"); END;
        CREATE TRIGGER IF NOT EXISTS declared_graph_manifest_events_no_delete
            BEFORE DELETE ON declared_graph_manifest_events
            BEGIN SELECT RAISE(ABORT, \"declared graph manifest history is append only\"); END;",
    )?;

    // ── v59 (#1173): non-authoritative experience projections ─────────────
    // A current projection is a compact cache over canonical entity IDs and
    // accepted serving/preload telemetry. The normalized source table makes
    // dependency invalidation exact; the ledger records rebuild inputs without
    // retaining bodies, prompts, credentials, or authority material.
    conn.execute_batch(
    "CREATE TABLE IF NOT EXISTS experience_projections (
        projection_id TEXT PRIMARY KEY,
        schema_version INTEGER NOT NULL,
        projection_version INTEGER NOT NULL,
        projection_revision INTEGER NOT NULL DEFAULT 1,
        experience_id TEXT NOT NULL,
        tenant_id TEXT NOT NULL,
        workspace_hash TEXT NOT NULL,
        principal_id TEXT NOT NULL,
        agent_id TEXT NOT NULL DEFAULT '',
        graph_side TEXT NOT NULL,
        layer TEXT NOT NULL,
        source_event_ids_json TEXT NOT NULL DEFAULT '[]',
        pulse_ids_json TEXT NOT NULL DEFAULT '[]',
        activation REAL NOT NULL DEFAULT 0.0,
        utility REAL NOT NULL DEFAULT 0.0,
        preference REAL NOT NULL DEFAULT 0.0,
        confidence REAL NOT NULL DEFAULT 0.0,
        source_digest TEXT NOT NULL,
        projection_digest TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'active',
        state_reason TEXT NOT NULL DEFAULT '',
        observed_at_unix_ms INTEGER NOT NULL,
        built_at_unix_ms INTEGER NOT NULL,
        updated_at_unix_ms INTEGER NOT NULL,
        UNIQUE(tenant_id, workspace_hash, principal_id, experience_id)
     );
     CREATE INDEX IF NOT EXISTS idx_experience_projections_source_state
        ON experience_projections(state, updated_at_unix_ms);
     CREATE TABLE IF NOT EXISTS experience_projection_sources (
        projection_id TEXT NOT NULL REFERENCES experience_projections(projection_id) ON DELETE CASCADE,
        source_entity_id TEXT NOT NULL,
        source_digest TEXT NOT NULL,
        PRIMARY KEY(projection_id, source_entity_id)
     );
     CREATE INDEX IF NOT EXISTS idx_experience_projection_sources_entity
        ON experience_projection_sources(source_entity_id);
     CREATE TABLE IF NOT EXISTS experience_projection_events (
        event_id TEXT PRIMARY KEY,
        projection_id TEXT NOT NULL REFERENCES experience_projections(projection_id) ON DELETE CASCADE,
        event_kind TEXT NOT NULL,
        source_entity_ids_json TEXT NOT NULL DEFAULT '[]',
        source_event_ids_json TEXT NOT NULL DEFAULT '[]',
        pulse_ids_json TEXT NOT NULL DEFAULT '[]',
        source_digest TEXT NOT NULL,
        projection_digest TEXT NOT NULL,
        tenant_id TEXT NOT NULL,
        workspace_hash TEXT NOT NULL,
        principal_id TEXT NOT NULL,
        agent_id TEXT NOT NULL DEFAULT '',
        recorded_at_unix_ms INTEGER NOT NULL,
        UNIQUE(projection_id, event_kind, source_digest, source_event_ids_json, pulse_ids_json)
     );
     CREATE INDEX IF NOT EXISTS idx_experience_projection_events_scope
        ON experience_projection_events(workspace_hash, principal_id, recorded_at_unix_ms);",
    )?;

    // ── v60 (#1182): scoped task-state projection ────────────────────────
    // This is a rebuildable serving projection, not canonical memory/history.
    // Its JSON payload is restricted by task_state::TaskState validation to
    // bounded metadata, IDs, and digests; entity bodies and prompts remain in
    // their canonical stores and are re-read through governed readers.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS task_state_projections (
           task_id TEXT NOT NULL,
           tenant_id TEXT NOT NULL,
           workspace_hash TEXT NOT NULL,
           principal_id TEXT NOT NULL,
           agent_id TEXT NOT NULL,
           schema_version TEXT NOT NULL,
           state_sequence INTEGER NOT NULL,
           base_sequence INTEGER NOT NULL,
           observed_input_digest TEXT NOT NULL,
           source_digest TEXT NOT NULL,
           evidence_digest TEXT NOT NULL,
           state_digest TEXT NOT NULL,
           state_json TEXT NOT NULL,
           created_at_unix_ms INTEGER NOT NULL,
           updated_at_unix_ms INTEGER NOT NULL,
           PRIMARY KEY (tenant_id, workspace_hash, principal_id, agent_id, task_id)
       );
       CREATE INDEX IF NOT EXISTS idx_task_state_projections_scope
           ON task_state_projections(workspace_hash, principal_id, agent_id, updated_at_unix_ms);",
    )?;

    // Stamp the migration level so subsequent opens skip the probe block above.
    conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;

    ensure_column(conn, "provider_sources", "author", "TEXT")?;
    ensure_column(conn, "provider_source_events", "author", "TEXT")?;
    Ok(())
}

/// Check if a database has the v0.2.0 entities table.
#[allow(dead_code)]
pub fn is_v0_2_0(conn: &Connection) -> Result<bool, Box<dyn std::error::Error>> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='entities'",
        [],
        |r| r.get(0),
    )?;
    Ok(count > 0)
}

/// Check if a database has the old v0.1.x memories table.
pub fn has_v0_1_memories(conn: &Connection) -> Result<bool, Box<dyn std::error::Error>> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories'",
        [],
        |r| r.get(0),
    )?;
    Ok(count > 0)
}

/// Get total entity count.
#[allow(dead_code)]
pub fn entity_count(conn: &Connection) -> Result<i64, Box<dyn std::error::Error>> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))?;
    Ok(count)
}

/// Truncate `s` to at most `max_bytes` bytes without splitting a UTF-8
/// character. `&s[..n]` panics when `n` is not a char boundary, so walk the
/// cut point back to the nearest boundary instead (stable-Rust equivalent of
/// the nightly `floor_char_boundary`). (#352)
fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn migration_key_for_id(base: &str, id: &str, collides: bool) -> String {
    if !collides {
        return base.to_string();
    }
    // Preserve the historical compact key when it is unique, but make the
    // collision case deterministic and lossless. The full SHA-256 suffix is
    // derived from the source id, so two legacy ids sharing the 20-byte prefix
    // cannot silently overwrite one another through INSERT OR REPLACE.
    format!("{base}-{}", crate::trust_admission::digest_text(id))
}

/// Migrate from v0.1.x schema to v0.2.0.
///
/// Opens the old DB, reads all memories, writes them as entities into the new schema,
/// and returns a MigrationReport.
pub fn migrate_from_v0_1(
    old_path: &str,
    conn: &Connection,
) -> Result<MigrationReport, Box<dyn std::error::Error>> {
    migrate_from_v0_1_with_fts(old_path, conn, true, None)
}

/// Import legacy rows without copying their body text into FTS. Mode-aware
/// callers (the `Database` wrapper) rebuild FTS after the import so an
/// encrypted target never has a plaintext FTS window.
pub(crate) fn migrate_from_v0_1_without_fts(
    old_path: &str,
    conn: &Connection,
) -> Result<MigrationReport, Box<dyn std::error::Error>> {
    migrate_from_v0_1_with_fts(old_path, conn, false, None)
}

/// Import legacy rows directly into an encrypted target. Canonical bodies are
/// encrypted before they are inserted and FTS is deliberately left empty until
/// the Database wrapper performs the keyed rebuild.
pub(crate) fn migrate_from_v0_1_with_encryption(
    old_path: &str,
    conn: &Connection,
    encryption: &EncryptionManager,
) -> Result<MigrationReport, Box<dyn std::error::Error>> {
    migrate_from_v0_1_with_fts(old_path, conn, false, Some(encryption))
}

fn migrate_from_v0_1_with_fts(
    old_path: &str,
    conn: &Connection,
    populate_fts: bool,
    encryption: Option<&EncryptionManager>,
) -> Result<MigrationReport, Box<dyn std::error::Error>> {
    let old_conn = Connection::open(old_path)?;

    if !has_v0_1_memories(&old_conn)? {
        return Ok(MigrationReport {
            total_old_memories: 0,
            entities_created: 0,
            entities_updated: 0,
            errors: vec!["No v0.1.x memories table found in source DB".to_string()],
            completed_at_unix_ms: now_ms(),
        });
    }

    // Ensure target has v0.2.0 schema
    initialize_schema(conn)?;
    // Import and any optional plaintext FTS population are one transaction so
    // an interrupted keyed migration cannot expose a partially written target.
    let tx = conn.unchecked_transaction()?;

    let mut stmt = old_conn.prepare(
        "SELECT id, content, type, summary, relevance, decay_score, retrieval_count,
                layer, topic_path, created_at_unix_ms, last_accessed_unix_ms,
                workspace_hash, tags, links, source, verified
         FROM memories",
    )?;

    let old_memories: Vec<(
        String,
        String,
        String,
        Option<String>,
        f64,
        f64,
        i64,
        String,
        String,
        i64,
        i64,
        String,
        String,
        String,
        String,
        i32,
    )> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,         // id
                row.get::<_, String>(1)?,         // content
                row.get::<_, String>(2)?,         // type
                row.get::<_, Option<String>>(3)?, // summary
                row.get::<_, f64>(4)?,            // relevance
                row.get::<_, f64>(5)?,            // decay_score
                row.get::<_, i64>(6)?,            // retrieval_count
                row.get::<_, String>(7)?,         // layer
                row.get::<_, String>(8)?,         // topic_path
                row.get::<_, i64>(9)?,            // created_at_unix_ms
                row.get::<_, i64>(10)?,           // last_accessed_unix_ms
                row.get::<_, String>(11)?,        // workspace_hash
                row.get::<_, String>(12)?,        // tags
                row.get::<_, String>(13)?,        // links
                row.get::<_, String>(14)?,        // source
                row.get::<_, i32>(15)?,           // verified
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);

    // Read the entire source before mutating the target. A malformed row must
    // abort the migration rather than allowing a partial import to commit.
    let total = old_memories.len() as i64;
    let mut base_key_counts = std::collections::HashMap::<(String, String), usize>::new();
    for row in &old_memories {
        let base = format!("migrated-{}", truncate_at_char_boundary(&row.0, 20));
        *base_key_counts.entry((row.11.clone(), base)).or_insert(0) += 1;
    }

    let mut created = 0i64;
    let mut updated = 0i64;

    for (
        id,
        content,
        mem_type,
        summary,
        _relevance,
        decay_score,
        retrieval_count,
        layer,
        topic_path,
        created_at,
        last_accessed,
        workspace_hash,
        tags_str,
        links_str,
        source,
        verified,
    ) in old_memories
    {
        // Build body_json: wrap content + summary. Serialization errors are
        // import errors and therefore roll back the whole target transaction.
        let body = serde_json::to_string(&json!({
            "content": content,
            "summary": summary.unwrap_or_default(),
        }))?;

        // Parse tags as JSON and inject workspace_hash if present. Preserve
        // the legacy value when it is valid JSON; malformed source data is not
        // silently discarded.
        let mut tags_value: serde_json::Value = serde_json::from_str(&tags_str)
            .map_err(|e| format!("invalid tags JSON for legacy id {id}: {e}"))?;
        if !workspace_hash.is_empty() {
            if let Some(arr) = tags_value.as_array_mut() {
                arr.push(json!(format!("workspace:{}", workspace_hash)));
            }
        }
        let tags_json = serde_json::to_string(&tags_value)?;

        let category = "general".to_string();
        let base_key = format!("migrated-{}", truncate_at_char_boundary(&id, 20));
        let key = migration_key_for_id(
            &base_key,
            &id,
            base_key_counts
                .get(&(workspace_hash.clone(), base_key.clone()))
                .copied()
                .unwrap_or(0)
                > 1,
        );

        let verified_int = if verified != 0 { 1 } else { 0 };
        let aad = crate::db::Database::build_aad(&category, &key);
        let stored_body = if let Some(enc) = encryption {
            enc.encrypt(&body, aad.as_bytes())?
        } else {
            body.clone()
        };
        let stored_hints = if let Some(enc) = encryption {
            enc.encrypt("[]", aad.as_bytes())?
        } else {
            "[]".to_string()
        };

        // `INSERT OR REPLACE` remains idempotent for the same source id, but
        // it must never replace a different target identity after truncation
        // or when importing into a non-empty destination.
        let existing_id: Option<String> = tx
            .query_row(
                "SELECT id FROM entities
                 WHERE category = ?1 AND key = ?2 AND workspace_hash = ?3",
                params![category, key, workspace_hash],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(existing_id) = existing_id {
            if existing_id != id {
                return Err(format!(
                    "legacy migration identity collision for category=general key={key} workspace={workspace_hash}"
                )
                .into());
            }
            updated += 1;
        } else {
            created += 1;
        }

        tx.execute(
            "INSERT OR REPLACE INTO entities
             (id, category, key, body_json, status, type, tags,
              decay_score, retrieval_count, layer, topic_path,
              archived, archive_reason, links, verified, source, workspace_hash,
              hints, created_at_unix_ms, last_accessed_unix_ms)
             VALUES (?1, ?2, ?3, ?4, 'active', ?5, ?6,
                     ?7, ?8, ?9, ?10,
                     0, '', ?11, ?12, ?13, ?14,
                     ?15, ?16, ?17)",
            params![
                id,
                category,
                key,
                stored_body,
                mem_type,
                tags_json,
                decay_score,
                retrieval_count,
                layer,
                topic_path,
                links_str,
                verified_int,
                source,
                workspace_hash,
                stored_hints,
                created_at,
                last_accessed,
            ],
        )?;
    }

    // Plaintext callers retain the historical FTS population behavior. The
    // Database wrapper uses the no-FTS variant and owns the mode-aware rebuild.
    tx.execute("DELETE FROM entities_fts", [])?;
    if populate_fts {
        tx.execute(
            "INSERT INTO entities_fts (rowid, body_json)
             SELECT rowid, body_json FROM entities WHERE archived = 0",
            [],
        )?;
    }
    tx.commit()?;

    Ok(MigrationReport {
        total_old_memories: total,
        entities_created: created,
        entities_updated: updated,
        errors: Vec::new(),
        completed_at_unix_ms: now_ms(),
    })
}

/// Gather database statistics across all tables.
pub fn gather_stats(conn: &Connection, db_path: &str) -> Result<Stats, Box<dyn std::error::Error>> {
    let total_entities: i64 = conn.query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))?;
    // #493: active/archived split. Every user-facing read path (list_entities,
    // count_entities, recall) filters archived = 0, so stats must expose the
    // same view; total_entities stays archived-inclusive for compatibility.
    let active_entities: i64 = conn.query_row(
        "SELECT COUNT(*) FROM entities WHERE archived = 0",
        [],
        |r| r.get(0),
    )?;
    let archived_entities = total_entities - active_entities;

    let by_category = query_grouped_counts(conn, "entities", "category", "")?;
    let by_type = query_grouped_counts(conn, "entities", "type", "")?;
    let by_layer = query_grouped_counts(conn, "entities", "layer", "")?;
    let by_category_active =
        query_grouped_counts(conn, "entities", "category", "WHERE archived = 0")?;
    let by_type_active = query_grouped_counts(conn, "entities", "type", "WHERE archived = 0")?;
    let by_layer_active = query_grouped_counts(conn, "entities", "layer", "WHERE archived = 0")?;

    let total_journal: i64 = conn.query_row("SELECT COUNT(*) FROM journal", [], |r| r.get(0))?;

    let total_state: i64 = conn.query_row("SELECT COUNT(*) FROM state", [], |r| r.get(0))?;

    let db_size = std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0);

    let oldest: Option<i64> = conn
        .query_row("SELECT MIN(created_at_unix_ms) FROM entities", [], |r| {
            r.get(0)
        })
        .ok();
    let newest: Option<i64> = conn
        .query_row("SELECT MAX(created_at_unix_ms) FROM entities", [], |r| {
            r.get(0)
        })
        .ok();

    // Graph health (#365): community membership + modularity, so operators
    // can see whether the link graph has global structure. Defaults keep the
    // stats call working even if the communities table is somehow absent.
    let total_communities: i64 = conn
        .query_row("SELECT COUNT(*) FROM communities", [], |r| r.get(0))
        .unwrap_or(0);
    let graph_modularity: Option<f64> = conn
        .query_row(
            "SELECT modularity FROM communities \
             ORDER BY generated_at_unix_ms DESC, id ASC LIMIT 1",
            [],
            |r| r.get(0),
        )
        .ok();

    // History growth visibility (#398): entity_history is append-only unless
    // retention/purge trims it, so surface size + the hot keys directly in
    // stats — rows, stored body bytes, and the top-10 keys by version count.
    let (total_history_rows, history_bytes): (i64, i64) = conn
        .query_row(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(body_json)), 0) FROM entity_history",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap_or((0, 0));
    let top_history_keys: serde_json::Value = {
        let mut items = Vec::new();
        if let Ok(mut stmt) = conn.prepare(
            "SELECT category, key, COUNT(*) AS versions, COALESCE(SUM(LENGTH(body_json)), 0) \
             FROM entity_history GROUP BY category, key \
             ORDER BY versions DESC, category ASC, key ASC LIMIT 10",
        ) {
            if let Ok(rows) = stmt.query_map([], |r| {
                Ok(json!({
                    "category": r.get::<_, String>(0)?,
                    "key": r.get::<_, String>(1)?,
                    "versions": r.get::<_, i64>(2)?,
                    "bytes": r.get::<_, i64>(3)?,
                }))
            }) {
                items.extend(rows.flatten());
            }
        }
        serde_json::Value::Array(items)
    };

    Ok(Stats {
        total_entities,
        active_entities,
        archived_entities,
        by_category,
        by_type,
        by_layer,
        by_category_active,
        by_type_active,
        by_layer_active,
        total_journal_events: total_journal,
        total_state_entries: total_state,
        db_file_size_bytes: db_size,
        oldest_unix_ms: oldest,
        newest_unix_ms: newest,
        total_communities,
        graph_modularity,
        total_history_rows,
        history_bytes,
        top_history_keys,
    })
}

fn query_grouped_counts(
    conn: &Connection,
    table: &str,
    column: &str,
    where_sql: &str,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let sql = format!(
        "SELECT {}, COUNT(*) FROM {} {} GROUP BY {} ORDER BY COUNT(*) DESC",
        column, table, where_sql, column
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)
                .unwrap_or_else(|_| "(null)".to_string()),
            r.get::<_, i64>(1).unwrap_or(0),
        ))
    })?;

    let mut map = serde_json::Map::new();
    for (key, count) in rows.flatten() {
        map.insert(key, json!(count));
    }
    Ok(serde_json::Value::Object(map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn temp_db() -> (Connection, String) {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "perseus_vault-test-schema-{}.db",
            uuid::Uuid::new_v4()
        ));
        let path_str = path.to_str().unwrap().to_string();
        let conn = Connection::open(&path_str).expect("open test db");
        // #1031 follow-up (test-windows flake): raw fixture connections skip
        // the pool's PRAGMA init, so give them the same stall headroom the
        // pool fixtures get on Windows/macOS CI runners.
        #[cfg(any(windows, target_os = "macos"))]
        conn.execute_batch("PRAGMA busy_timeout=30000;")
            .expect("set fixture busy timeout");
        (conn, path_str)
    }

    #[test]
    fn initializes_schema_on_new_db() {
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("init schema");
        assert!(is_v0_2_0(&conn).unwrap());
    }

    /// #1020 regression: fresh-store and migrated-store PHYSICAL column
    /// order must be identical. `entity_from_row` hydrates `SELECT *`
    /// projections positionally, so a column placed mid-DDL (instead of via
    /// the append-only migration) shifts every later column on fresh stores
    /// only and silently mis-hydrates rows there.
    #[test]
    fn fresh_and_migrated_stores_share_physical_column_order() {
        // Fresh store: full initialize_schema from scratch.
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("init schema");
        let fresh: Vec<String> = {
            let mut stmt = conn.prepare("PRAGMA table_info(entities)").unwrap();
            let rows = stmt.query_map([], |r| r.get::<_, String>(1)).unwrap();
            rows.map(|r| r.unwrap()).collect()
        };
        // Migrated store: legacy DDL only, stamped with an old version, then
        // migrated — the upgrade path a real pre-#1020 store takes.
        let (conn2, _path2) = temp_db();
        conn2.execute_batch(DDL_V0_2_0).unwrap();
        conn2.pragma_update(None, "user_version", 10).unwrap();
        initialize_schema(&conn2).expect("migrate legacy store");
        let migrated: Vec<String> = {
            let mut stmt = conn2.prepare("PRAGMA table_info(entities)").unwrap();
            let rows = stmt.query_map([], |r| r.get::<_, String>(1)).unwrap();
            rows.map(|r| r.unwrap()).collect()
        };
        assert_eq!(
            fresh, migrated,
            "fresh and migrated stores must share physical column order"
        );
        assert!(
            fresh.iter().any(|c| c == "fingerprint"),
            "fingerprint column missing from {fresh:?}"
        );
    }

    #[test]
    fn stamps_user_version_and_gates_migration_probes() {
        // Fresh init stamps the current schema version.
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("init schema");
        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            v, SCHEMA_VERSION,
            "fresh init must stamp the schema version"
        );

        // Re-running on an already-current DB is a no-op that preserves data and
        // leaves the version untouched (the probe block is skipped).
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('v-test', 'insight', 'k', '{}', 0, 0)",
            [],
        )
        .unwrap();
        initialize_schema(&conn).expect("re-init should be a no-op");
        let v2: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v2, SCHEMA_VERSION);
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM entities WHERE id='v-test'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, 1, "re-init must not drop data");
    }

    #[test]
    fn migrates_pre_versioned_db_missing_a_column() {
        // Simulate a legacy DB at user_version=0 that predates the visibility
        // column: the gate must still run the probes and add the column, then
        // stamp the version so later opens skip.
        let (conn, _path) = temp_db();
        // Base v0.2.0 columns the DDL's indexes reference, but WITHOUT the
        // later ALTER-added columns (embedding/always_on/certainty/
        // workspace_hash/agent_id/visibility, journal agent_id/audit_hash).
        conn.execute_batch(
            "CREATE TABLE entities (
                id TEXT PRIMARY KEY, category TEXT NOT NULL DEFAULT 'general', key TEXT NOT NULL,
                body_json TEXT NOT NULL DEFAULT '{}', archived INTEGER DEFAULT 0,
                retrieval_count INTEGER DEFAULT 0,
                created_at_unix_ms INTEGER NOT NULL, last_accessed_unix_ms INTEGER NOT NULL
             );
             CREATE TABLE journal (
                id TEXT PRIMARY KEY, entity_id TEXT DEFAULT '',
                created_at_unix_ms INTEGER NOT NULL
             );",
        )
        .unwrap();
        assert!(
            conn.prepare("SELECT visibility FROM entities LIMIT 1")
                .is_err(),
            "precondition: legacy table lacks visibility"
        );

        initialize_schema(&conn).expect("migrate legacy db");

        assert!(
            conn.prepare("SELECT visibility FROM entities LIMIT 1")
                .is_ok(),
            "visibility column must be added during gated migration"
        );
        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
    }

    #[test]
    fn apply_migrations_post_lock_version_recheck_is_noop() {
        // #353: the loser of the migration race blocks on BEGIN IMMEDIATE,
        // then must see the winner's stamped user_version and skip cleanly.
        // Exercise that re-check path directly: on an already-current DB,
        // apply_migrations must return Ok without touching anything.
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("init schema");
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('recheck-test', 'insight', 'k', '{}', 0, 0)",
            [],
        )
        .unwrap();

        apply_migrations(&conn).expect("post-lock re-check must be a clean no-op");

        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn ensure_column_treats_duplicate_column_as_benign() {
        // #353 defense-in-depth: simulate losing the probe/ALTER race — the
        // column appears after our probe would have run. ensure_column's
        // ALTER error path must swallow "duplicate column name".
        let (conn, _path) = temp_db();
        conn.execute_batch("CREATE TABLE t (id TEXT PRIMARY KEY, extra TEXT);")
            .unwrap();
        // Probe path: column exists, no ALTER attempted.
        ensure_column(&conn, "t", "extra", "TEXT").expect("existing column is a no-op");
        // Error path: bypass the probe by asserting the raw duplicate error is
        // matched by the same predicate ensure_column uses.
        let err = conn
            .execute_batch("ALTER TABLE t ADD COLUMN extra TEXT;")
            .unwrap_err();
        assert!(
            err.to_string().contains("duplicate column name"),
            "SQLite duplicate-column error text changed: {err}"
        );
    }

    #[test]
    fn v27_migration_adds_epistemic_state_defaulting_to_candidate() {
        // #880: a v26-era store (full column set minus epistemic_state) must
        // gain the trust axis on migration, and pre-existing rows must read
        // as 'candidate' — the fail-closed reading for records written before
        // admission evidence existed.
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("fresh init");
        // Simulate a v26 store: drop the v27 column, stamp the old version.
        conn.execute_batch("ALTER TABLE entities DROP COLUMN epistemic_state;")
            .unwrap();
        conn.pragma_update(None, "user_version", 26i64).unwrap();
        assert!(
            conn.prepare("SELECT epistemic_state FROM entities LIMIT 1")
                .is_err(),
            "precondition: v26 store lacks epistemic_state"
        );
        // A legacy row written before the axis existed.
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('legacy', 'insight', 'k', '{}', 0, 0)",
            [],
        )
        .unwrap();

        // Migrate.
        initialize_schema(&conn).expect("migrate legacy store to current");

        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            v, SCHEMA_VERSION,
            "migration must stamp the current version"
        );
        let state: String = conn
            .query_row(
                "SELECT epistemic_state FROM entities WHERE id = 'legacy'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(state, "candidate", "legacy rows default to candidate");
        // Fresh writes on the migrated store still land with the column.
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('fresh', 'insight', 'k2', '{}', 0, 0)",
            [],
        )
        .unwrap();
        let fresh: String = conn
            .query_row(
                "SELECT epistemic_state FROM entities WHERE id = 'fresh'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(fresh, "candidate");
    }

    #[test]
    fn concurrent_opens_of_pre_upgrade_db_both_succeed() {
        // #353 end-to-end: two "processes" (independent connections — same
        // lock semantics as separate OS processes) open the same pre-upgrade
        // DB and run initialize_schema concurrently. Before the fix the loser
        // could fail with "duplicate column name"; now BEGIN IMMEDIATE
        // serializes them and the loser's post-lock re-check no-ops.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "perseus_vault-test-race-{}.db",
            uuid::Uuid::new_v4()
        ));
        let path_str = path.to_str().unwrap().to_string();

        // Pre-upgrade fixture: legacy tables without the ALTER-added columns,
        // user_version=0 (same shape as migrates_pre_versioned_db_missing_a_column).
        // #1020 follow-up (Windows CI flake on main, 2026-08-13): the fixture
        // runs WAL like every production store — the DELETE-journaling default
        // here tested a mode production never uses, and its reader/writer
        // upgrade-deadlock class is irrelevant to the #353 contract.
        {
            let conn = Connection::open(&path_str).unwrap();
            conn.execute_batch(
                "CREATE TABLE entities (
                    id TEXT PRIMARY KEY, category TEXT NOT NULL DEFAULT 'general', key TEXT NOT NULL,
                    body_json TEXT NOT NULL DEFAULT '{}', archived INTEGER DEFAULT 0,
                    retrieval_count INTEGER DEFAULT 0,
                    created_at_unix_ms INTEGER NOT NULL, last_accessed_unix_ms INTEGER NOT NULL
                 );
                 CREATE TABLE journal (
                    id TEXT PRIMARY KEY, entity_id TEXT DEFAULT '',
                    created_at_unix_ms INTEGER NOT NULL
                 );
                 PRAGMA journal_mode=WAL;",
            )
            .unwrap();
        }

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let path = path_str.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let conn = Connection::open(&path).map_err(|e| e.to_string())?;
                    // The loser must WAIT on BEGIN IMMEDIATE, not fail fast
                    // with SQLITE_BUSY. 60s (not the pool's 5s default): on
                    // the shared Windows runner a cold-FS/Defender stall can
                    // push the winner's migration past 5s (#950 recorded a
                    // 411ms stall on a single first write; full-suite runs
                    // there take ~600s vs ~80s serial on Linux), and the
                    // migration here is bounded work — expiry is not part of
                    // the #353 contract.
                    conn.execute_batch("PRAGMA busy_timeout=60000;")
                        .map_err(|e| e.to_string())?;
                    barrier.wait();
                    initialize_schema(&conn).map_err(|e| e.to_string())?;
                    Ok::<(), String>(())
                })
            })
            .collect();
        for h in handles {
            h.join()
                .expect("thread panicked")
                .expect("concurrent initialize_schema must succeed in both openers");
        }

        // The DB came out fully migrated exactly once.
        let conn = Connection::open(&path_str).unwrap();
        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        assert!(conn.prepare("SELECT emb_sig FROM entities LIMIT 1").is_ok());
        drop(conn);
        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn migrates_unique_index_to_workspace_scoped_identity() {
        // v4 (#339): a v3-era DB with the two-column unique index and existing
        // rows must come out with the three-column index, the old index
        // dropped, and cross-workspace key collisions storable.
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("fresh init");
        // Rewind to the v3 state: old index back, new index gone, version 3.
        conn.execute_batch(
            "DROP INDEX IF EXISTS idx_entities_category_key_ws;
             CREATE UNIQUE INDEX idx_entities_category_key ON entities(category, key);
             PRAGMA user_version = 3;",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, workspace_hash, created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('mig-a', 'note', 'k', '{}', 'ws-alpha', 0, 0)",
            [],
        )
        .unwrap();

        initialize_schema(&conn).expect("v3 -> v4 migration");

        let old_idx: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_entities_category_key'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(old_idx, 0, "old two-column unique index must be dropped");
        let new_idx: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_entities_category_key_ws'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(new_idx, 1, "workspace-scoped unique index must exist");

        // Same (category, key) in a different workspace now inserts…
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, workspace_hash, created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('mig-b', 'note', 'k', '{}', 'ws-beta', 0, 0)",
            [],
        )
        .expect("cross-workspace key collision must be storable after v4");
        // …while a true duplicate in the SAME workspace is still rejected.
        assert!(
            conn.execute(
                "INSERT INTO entities (id, category, key, body_json, workspace_hash, created_at_unix_ms, last_accessed_unix_ms)
                 VALUES ('mig-c', 'note', 'k', '{}', 'ws-alpha', 0, 0)",
                [],
            )
            .is_err(),
            "same-workspace duplicate must still violate uniqueness"
        );
    }

    #[test]
    fn migration_backfills_embedding_signatures() {
        // v6: embeddings stored before emb_sig existed must get a signature
        // during the gated migration, matching what store_embedding writes.
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("fresh init");
        // Rewind: drop the column's data by simulating a pre-v6 row.
        let emb: Vec<f32> = vec![1.0, -2.0, 0.5, -0.1];
        let blob: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, embedding, emb_sig,
                                   created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('sig-1', 'note', 'k', '{}', ?1, NULL, 0, 0)",
            params![blob],
        )
        .unwrap();
        conn.pragma_update(None, "user_version", 5).unwrap();

        initialize_schema(&conn).expect("v5 -> v6 migration");

        let sig: Vec<u8> = conn
            .query_row("SELECT emb_sig FROM entities WHERE id = 'sig-1'", [], |r| {
                r.get(0)
            })
            .expect("emb_sig must be backfilled");
        assert_eq!(sig, crate::db::embedding_signature(&emb));
    }

    #[test]
    fn adds_bitemporal_columns_and_backfills_recorded_at() {
        // A legacy DB (no bi-temporal columns) with one row predating the migration.
        let (conn, _path) = temp_db();
        conn.execute_batch(
            "CREATE TABLE entities (
                id TEXT PRIMARY KEY, category TEXT NOT NULL DEFAULT 'general', key TEXT NOT NULL,
                body_json TEXT NOT NULL DEFAULT '{}', archived INTEGER DEFAULT 0,
                retrieval_count INTEGER DEFAULT 0,
                created_at_unix_ms INTEGER NOT NULL, last_accessed_unix_ms INTEGER NOT NULL
             );
             CREATE TABLE journal (
                id TEXT PRIMARY KEY, entity_id TEXT DEFAULT '',
                created_at_unix_ms INTEGER NOT NULL
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('e1', 'general', 'k', '{}', 111, 222)",
            [],
        )
        .unwrap();
        assert!(
            conn.prepare("SELECT recorded_at_unix_ms FROM entities LIMIT 1")
                .is_err(),
            "precondition: legacy table lacks the bi-temporal columns"
        );

        initialize_schema(&conn).expect("migrate legacy db to bi-temporal schema");

        // All six bi-temporal columns must now exist.
        for col in [
            "valid_from_unix_ms",
            "valid_to_unix_ms",
            "recorded_at_unix_ms",
            "invalidated_at_unix_ms",
            "supersedes",
            "superseded_by",
        ] {
            assert!(
                conn.prepare(&format!("SELECT {col} FROM entities LIMIT 1"))
                    .is_ok(),
                "column {col} must be added during migration"
            );
        }

        // recorded_at backfilled to created_at; the row is live (not invalidated)
        // and unbounded in valid time — i.e. unchanged in meaning.
        let recorded: i64 = conn
            .query_row(
                "SELECT recorded_at_unix_ms FROM entities WHERE id='e1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(recorded, 111, "recorded_at must backfill to created_at");
        let invalidated: Option<i64> = conn
            .query_row(
                "SELECT invalidated_at_unix_ms FROM entities WHERE id='e1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            invalidated, None,
            "existing rows must be live (not invalidated)"
        );
        // v7 (#363): the historical "NULL = valid since recorded" convention is
        // made explicit — valid_from backfills to the transaction time.
        let valid_from: Option<i64> = conn
            .query_row(
                "SELECT valid_from_unix_ms FROM entities WHERE id='e1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            valid_from,
            Some(111),
            "v7 must backfill valid_from to recorded_at"
        );
        let valid_to: Option<i64> = conn
            .query_row(
                "SELECT valid_to_unix_ms FROM entities WHERE id='e1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(valid_to, None, "valid_to must stay NULL (= still true)");

        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
    }

    #[test]
    fn v7_valid_from_backfill_is_idempotent_and_preserves_explicit_values() {
        // A v6-era DB: bi-temporal columns exist, valid_from is NULL on one row
        // and explicitly set on another. The v7 backfill must fill only the
        // NULL row, leave the explicit one alone, and be safely re-runnable.
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("fresh init");
        conn.execute_batch(
            "INSERT INTO entities (id, category, key, body_json, recorded_at_unix_ms,
                                   valid_from_unix_ms, created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('v7-null', 'note', 'a', '{}', 500, NULL, 400, 400),
                    ('v7-set',  'note', 'b', '{}', 500, 42,   400, 400);
             INSERT INTO entity_history (history_id, id, category, key, body_json,
                                         recorded_at_unix_ms, valid_from_unix_ms,
                                         created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('h1', 'v7-null', 'note', 'a', '{}', 300, NULL, 300, 300);
             PRAGMA user_version = 6;",
        )
        .unwrap();

        initialize_schema(&conn).expect("v6 -> v7 migration");

        let vf_null: Option<i64> = conn
            .query_row(
                "SELECT valid_from_unix_ms FROM entities WHERE id='v7-null'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            vf_null,
            Some(500),
            "NULL valid_from must backfill to recorded_at"
        );
        let vf_set: Option<i64> = conn
            .query_row(
                "SELECT valid_from_unix_ms FROM entities WHERE id='v7-set'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(vf_set, Some(42), "explicit valid_from must be preserved");
        let vf_hist: Option<i64> = conn
            .query_row(
                "SELECT valid_from_unix_ms FROM entity_history WHERE history_id='h1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(vf_hist, Some(300), "history rows must backfill too");

        // Re-run (rewind the stamp): idempotent, values unchanged.
        conn.pragma_update(None, "user_version", 6).unwrap();
        initialize_schema(&conn).expect("v7 re-run");
        let vf_again: Option<i64> = conn
            .query_row(
                "SELECT valid_from_unix_ms FROM entities WHERE id='v7-null'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(vf_again, Some(500));
        let vf_set_again: Option<i64> = conn
            .query_row(
                "SELECT valid_from_unix_ms FROM entities WHERE id='v7-set'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(vf_set_again, Some(42));
    }

    #[test]
    fn fresh_db_has_bitemporal_columns_and_live_index() {
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("init schema");
        assert!(conn
            .prepare("SELECT invalidated_at_unix_ms FROM entities LIMIT 1")
            .is_ok());
        let idx: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_entities_invalidated'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            idx, 1,
            "idx_entities_invalidated should be created on a fresh DB"
        );
    }

    #[test]
    fn creates_recall_ranking_index() {
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("init schema");
        // Index must exist...
        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_entities_recall'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(exists, 1, "idx_entities_recall should be created");
        // ...and the recall browse query must use it with NO temp-b-tree sort.
        // Uses the FULL 3-key ordering the recall path actually emits (db.rs:
        // "ORDER BY retrieval_count DESC, last_accessed_unix_ms DESC, id ASC"),
        // including the #254 determinism tie-break. This is the v13 guard: the
        // pre-v13 index covered only the first two keys, so this exact query
        // needed a TEMP B-TREE to order the tie-break — the O(tie-group) sort
        // that made browse ~30ms at 1M rows.
        let plan: Vec<String> = conn
            .prepare(
                "EXPLAIN QUERY PLAN SELECT id FROM entities WHERE archived = 0 \
                 ORDER BY retrieval_count DESC, last_accessed_unix_ms DESC, id ASC LIMIT 20",
            )
            .unwrap()
            .query_map([], |r| r.get::<_, String>(3))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        let joined = plan.join(" | ");
        assert!(
            joined.contains("idx_entities_recall"),
            "recall query should use idx_entities_recall, got: {joined}"
        );
        assert!(
            !joined.to_uppercase().contains("TEMP B-TREE"),
            "recall browse (incl. id tie-break) should not need a temp-b-tree sort, got: {joined}"
        );
    }

    #[test]
    fn fresh_db_has_communities_table() {
        // v8 (#365): GraphRAG communities table + workspace index.
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("init schema");
        assert!(
            conn.prepare(
                "SELECT id, workspace_hash, member_ids, member_digest, summary, \
                          summary_entity_id, algorithm, modularity, member_count, \
                          generated_at_unix_ms FROM communities LIMIT 1"
            )
            .is_ok(),
            "communities table with all v8 columns must exist on a fresh DB"
        );
        let idx: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_communities_ws'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(idx, 1, "idx_communities_ws must be created");
    }

    #[test]
    fn migrates_legacy_db_to_v8_with_communities_table() {
        // A v6-era DB (fully migrated through emb_sig, user_version=6) must
        // gain the communities table when re-opened, and land on the current
        // SCHEMA_VERSION, without disturbing existing data.
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("fresh init");
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('v8-keep', 'note', 'k', '{}', 0, 0)",
            [],
        )
        .unwrap();
        // Rewind to the v6 state: communities table gone, version 6.
        conn.execute_batch(
            "DROP INDEX IF EXISTS idx_communities_ws;
             DROP TABLE IF EXISTS communities;
             PRAGMA user_version = 6;",
        )
        .unwrap();
        assert!(
            conn.prepare("SELECT id FROM communities LIMIT 1").is_err(),
            "precondition: v6 DB lacks the communities table"
        );

        initialize_schema(&conn).expect("v6 -> v8 migration");

        assert!(conn.prepare("SELECT id FROM communities LIMIT 1").is_ok());
        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        let kept: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM entities WHERE id='v8-keep'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(kept, 1, "migration must not drop data");
    }

    #[test]
    fn migrates_legacy_db_to_v10_with_dedup_signatures() {
        // v10 (#392): a v9-era DB must gain the dedup_signatures side table on
        // reopen, land on the current SCHEMA_VERSION, and keep its data. No
        // eager backfill: pre-existing entities simply have no signature row
        // (the dedup scan rebuilds and writes back lazily).
        let (conn, _path) = temp_db();
        initialize_schema(&conn).expect("fresh init");
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, last_accessed_unix_ms)
             VALUES ('v10-keep', 'note', 'k', '{}', 0, 0)",
            [],
        )
        .unwrap();
        // Rewind to the v9 state: side tables gone, version 9.
        conn.execute_batch(
            "DROP TABLE IF EXISTS dedup_signatures;
             DROP TABLE IF EXISTS dedup_signature_blobs;
             PRAGMA user_version = 9;",
        )
        .unwrap();
        assert!(
            conn.prepare("SELECT entity_id FROM dedup_signatures LIMIT 1")
                .is_err(),
            "precondition: v9 DB lacks the dedup_signatures table"
        );

        initialize_schema(&conn).expect("v9 -> v10 migration");

        assert!(
            conn.prepare(
                "SELECT entity_id, body_len, body_hash, tg_count, histo \n                 FROM dedup_signatures LIMIT 1"
            )
            .is_ok(),
            "dedup_signatures with all v10 columns must exist"
        );
        assert!(
            conn.prepare("SELECT entity_id, sig FROM dedup_signature_blobs LIMIT 1")
                .is_ok(),
            "dedup_signature_blobs must exist"
        );
        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        let kept: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM entities WHERE id='v10-keep'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(kept, 1, "migration must not drop data");
        // v17 (#476): the migration EAGERLY backfills — "every active row has
        // a signature" is now the invariant the signature-driven scan relies
        // on (the pre-v17 lazy backfill rode the entities walk the scan no
        // longer does). The signature must describe the stored body exactly,
        // with the row's scope columns.
        let (blen, cat, ws): (i64, String, String) = conn
            .query_row(
                "SELECT body_len, category, workspace_hash FROM dedup_signatures \
                 WHERE entity_id='v10-keep'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .expect("v17 must eagerly backfill the legacy row's signature");
        assert_eq!(blen, 2, "signature must describe the stored body ('{{}}')");
        assert_eq!(cat, "note");
        assert_eq!(ws, "");
    }

    #[test]
    fn migrates_legacy_db_to_v11_with_journal_workspace_hash() {
        // v11 (#417): a v10-era journal table has no workspace_hash column. The
        // gated migration must ALTER it in, create the (category,key,workspace)
        // index, stamp SCHEMA_VERSION, and preserve existing rows (defaulting to
        // an empty workspace_hash).
        let (conn, _path) = temp_db();
        // A pre-v11 journal table: all v10 columns, but no workspace_hash.
        conn.execute_batch(
            "CREATE TABLE journal (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL DEFAULT 'decision',
                evaluated_json TEXT DEFAULT '{}',
                acted_json TEXT DEFAULT '{}',
                forward_json TEXT DEFAULT '{}',
                category TEXT DEFAULT '',
                key TEXT DEFAULT '',
                entity_id TEXT DEFAULT '',
                agent_id TEXT DEFAULT '',
                audit_hash TEXT DEFAULT '',
                created_at_unix_ms INTEGER NOT NULL
             );
             INSERT INTO journal (id, category, key, created_at_unix_ms)
                 VALUES ('jrn-old', 'facts', 'k', 1);
             PRAGMA user_version = 10;",
        )
        .unwrap();
        assert!(
            conn.prepare("SELECT workspace_hash FROM journal LIMIT 1")
                .is_err(),
            "precondition: v10 journal lacks workspace_hash"
        );

        initialize_schema(&conn).expect("v10 -> v11 migration");

        // Column added, index created, version stamped.
        assert!(
            conn.prepare("SELECT workspace_hash FROM journal LIMIT 1")
                .is_ok(),
            "workspace_hash column must be added"
        );
        let has_idx: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_journal_catkeyws'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(has_idx, 1, "purge-match index must be created");
        let v: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, SCHEMA_VERSION);
        // Legacy row preserved, defaulting to empty workspace_hash.
        let (kept, ws): (i64, String) = conn
            .query_row(
                "SELECT COUNT(*), COALESCE(MAX(workspace_hash), '') FROM journal WHERE id='jrn-old'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(kept, 1, "migration must not drop journal rows");
        assert_eq!(
            ws, "",
            "legacy journal rows default to empty workspace_hash"
        );
    }

    #[test]
    fn detects_v0_1_memories_table() {
        let (conn, _path) = temp_db();
        conn.execute_batch("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT);")
            .expect("create v0.1 memories");
        assert!(has_v0_1_memories(&conn).unwrap());
        assert!(!is_v0_2_0(&conn).unwrap());
    }

    #[test]
    fn migration_from_v0_1_preserves_data() {
        let (old_conn, old_path) = temp_db();
        old_conn
            .execute_batch(
                "CREATE TABLE memories (
                    id TEXT PRIMARY KEY, content TEXT NOT NULL,
                    type TEXT DEFAULT 'insight', summary TEXT DEFAULT '',
                    relevance REAL DEFAULT 0.0, decay_score REAL DEFAULT 1.0,
                    retrieval_count INTEGER DEFAULT 0, layer TEXT DEFAULT 'working',
                    topic_path TEXT DEFAULT '', created_at_unix_ms INTEGER NOT NULL,
                    last_accessed_unix_ms INTEGER NOT NULL, workspace_hash TEXT DEFAULT '',
                    tags TEXT DEFAULT '{}', links TEXT DEFAULT '[]', source TEXT DEFAULT 'perseus-vault',
                    verified INTEGER DEFAULT 0
                );",
            )
            .expect("create v0.1 schema");

        let now = now_ms();
        old_conn
            .execute(
                "INSERT INTO memories (id, content, type, created_at_unix_ms, last_accessed_unix_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params!["mem-test1", "Test content", "insight", now, now],
            )
            .expect("insert test memory");
        drop(old_conn);

        let (new_conn, _new_path) = temp_db();
        let report = migrate_from_v0_1(&old_path, &new_conn).expect("migrate");

        assert_eq!(report.total_old_memories, 1);
        assert_eq!(report.entities_created, 1);
        assert!(report.errors.is_empty());

        // Verify entity exists
        let count: i64 = new_conn
            .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);

        // Verify body_json contains original content
        let body: String = new_conn
            .query_row(
                "SELECT body_json FROM entities WHERE id = 'mem-test1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(body.contains("Test content"));

        // Cleanup old db
        let _ = std::fs::remove_file(&old_path);
    }

    #[test]
    fn migration_without_fts_does_not_copy_plaintext() {
        let (old_conn, old_path) = temp_db();
        old_conn
            .execute_batch(
                "CREATE TABLE memories (
                    id TEXT PRIMARY KEY, content TEXT NOT NULL,
                    type TEXT DEFAULT 'insight', summary TEXT DEFAULT '',
                    relevance REAL DEFAULT 0.0, decay_score REAL DEFAULT 1.0,
                    retrieval_count INTEGER DEFAULT 0, layer TEXT DEFAULT 'working',
                    topic_path TEXT DEFAULT '', created_at_unix_ms INTEGER NOT NULL,
                    last_accessed_unix_ms INTEGER NOT NULL, workspace_hash TEXT DEFAULT '',
                    tags TEXT DEFAULT '{}', links TEXT DEFAULT '[]', source TEXT DEFAULT 'perseus-vault',
                    verified INTEGER DEFAULT 0
                );",
            )
            .expect("create v0.1 schema");
        let marker = "migration-fts-marker";
        let now = now_ms();
        old_conn
            .execute(
                "INSERT INTO memories (id, content, type, created_at_unix_ms, last_accessed_unix_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params!["mem-no-fts", marker, "insight", now, now],
            )
            .expect("insert test memory");
        drop(old_conn);

        let (new_conn, new_path) = temp_db();
        let report = migrate_from_v0_1_without_fts(&old_path, &new_conn).expect("migrate");
        assert_eq!(report.entities_created, 1);
        let leaked: i64 = new_conn
            .query_row(
                "SELECT COUNT(*) FROM entities_fts_content WHERE c0 LIKE '%migration-fts-marker%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(leaked, 0, "mode-aware callers must own FTS rebuilding");

        let _ = std::fs::remove_file(&old_path);
        let _ = std::fs::remove_file(&new_path);
    }

    #[test]
    fn encrypted_v01_migration_encrypts_default_hints() {
        let (old_conn, old_path) = temp_db();
        old_conn
            .execute_batch(
                "CREATE TABLE memories (
                    id TEXT PRIMARY KEY, content TEXT NOT NULL,
                    type TEXT DEFAULT 'insight', summary TEXT DEFAULT '',
                    relevance REAL DEFAULT 0.0, decay_score REAL DEFAULT 1.0,
                    retrieval_count INTEGER DEFAULT 0, layer TEXT DEFAULT 'working',
                    topic_path TEXT DEFAULT '', created_at_unix_ms INTEGER NOT NULL,
                    last_accessed_unix_ms INTEGER NOT NULL, workspace_hash TEXT DEFAULT '',
                    tags TEXT DEFAULT '{}', links TEXT DEFAULT '[]', source TEXT DEFAULT 'perseus-vault',
                    verified INTEGER DEFAULT 0
                );",
            )
            .unwrap();
        old_conn
            .execute(
                "INSERT INTO memories (id, content, type, created_at_unix_ms, last_accessed_unix_ms)
                 VALUES (?1, ?2, ?3, 1, 1)",
                params!["encrypted-v01", "legacy encrypted body", "insight"],
            )
            .unwrap();
        drop(old_conn);

        let (new_conn, new_path) = temp_db();
        let key_path = std::env::temp_dir().join(format!(
            "perseus-vault-schema-key-{}.key",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&key_path, EncryptionManager::generate_key()).unwrap();
        let encryption = EncryptionManager::from_key_file(key_path.to_str().unwrap()).unwrap();
        migrate_from_v0_1_with_encryption(&old_path, &new_conn, &encryption).unwrap();

        let raw_body: String = new_conn
            .query_row(
                "SELECT body_json FROM entities WHERE id = 'encrypted-v01'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let raw_hints: String = new_conn
            .query_row(
                "SELECT hints FROM entities WHERE id = 'encrypted-v01'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_ne!(raw_body, r#"{"content":"legacy encrypted body"}"#);
        assert_ne!(raw_hints, "[]");
        let plaintext = match encryption.decrypt_body(
            &raw_body,
            crate::db::Database::build_aad("general", "migrated-encrypted-v01").as_bytes(),
        ) {
            crate::encryption::BodyDecrypt::Plaintext(value) => value,
            _ => panic!("migrated body must authenticate"),
        };
        assert!(plaintext.contains("legacy encrypted body"));
        let hints = match encryption.decrypt_body(
            &raw_hints,
            crate::db::Database::build_aad("general", "migrated-encrypted-v01").as_bytes(),
        ) {
            crate::encryption::BodyDecrypt::Plaintext(value) => value,
            _ => panic!("migrated hints must authenticate"),
        };
        assert_eq!(hints, "[]");

        let _ = std::fs::remove_file(&old_path);
        let _ = std::fs::remove_file(&new_path);
        let _ = std::fs::remove_file(&key_path);
    }

    #[test]
    fn truncate_at_char_boundary_never_splits_chars() {
        // Shorter than the limit: unchanged.
        assert_eq!(truncate_at_char_boundary("abc", 20), "abc");
        // Exactly at the limit: unchanged.
        assert_eq!(
            truncate_at_char_boundary("a".repeat(20).as_str(), 20),
            "a".repeat(20)
        );
        // ASCII over the limit: plain byte cut.
        assert_eq!(truncate_at_char_boundary("abcdef", 3), "abc");
        // Multi-byte char straddling the cut point: back up to the boundary.
        // "é" is 2 bytes; cutting "aé" at byte 2 lands mid-char.
        assert_eq!(truncate_at_char_boundary("aéz", 2), "a");
        // 4-byte char straddling the cut.
        assert_eq!(truncate_at_char_boundary("ab😀z", 4), "ab");
        // Degenerate limit 0.
        assert_eq!(truncate_at_char_boundary("é", 0), "");
    }

    #[test]
    fn future_schema_version_is_rejected_before_encryption_activation() {
        let conn = Connection::open_in_memory().expect("in-memory database");
        conn.pragma_update(None, "user_version", SCHEMA_VERSION + 1)
            .expect("set future schema version");
        let result = initialize_schema(&conn);
        assert!(
            result.is_err(),
            "a schema newer than this binary must fail closed"
        );
    }

    #[test]
    fn migration_from_v0_1_handles_multibyte_id_without_panic() {
        // #352: a legacy id whose multi-byte UTF-8 char straddles byte offset
        // 20 used to panic in the byte-index slice building `key`, aborting
        // the whole one-time migration. The char at bytes 19..21 ("é") is the
        // exact repro from the issue.
        let (old_conn, old_path) = temp_db();
        old_conn
            .execute_batch(
                "CREATE TABLE memories (
                    id TEXT PRIMARY KEY, content TEXT NOT NULL,
                    type TEXT DEFAULT 'insight', summary TEXT DEFAULT '',
                    relevance REAL DEFAULT 0.0, decay_score REAL DEFAULT 1.0,
                    retrieval_count INTEGER DEFAULT 0, layer TEXT DEFAULT 'working',
                    topic_path TEXT DEFAULT '', created_at_unix_ms INTEGER NOT NULL,
                    last_accessed_unix_ms INTEGER NOT NULL, workspace_hash TEXT DEFAULT '',
                    tags TEXT DEFAULT '{}', links TEXT DEFAULT '[]', source TEXT DEFAULT 'perseus-vault',
                    verified INTEGER DEFAULT 0
                );",
            )
            .expect("create v0.1 schema");

        // 19 ASCII bytes, then a 2-byte char occupying bytes 19..21.
        let evil_id = format!("{}é-tail", "x".repeat(19));
        assert!(
            !evil_id.is_char_boundary(20),
            "precondition: byte 20 is mid-char"
        );
        let now = now_ms();
        old_conn
            .execute(
                "INSERT INTO memories (id, content, type, created_at_unix_ms, last_accessed_unix_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![evil_id, "Unicode id content", "insight", now, now],
            )
            .expect("insert multibyte-id memory");
        drop(old_conn);

        let (new_conn, _new_path) = temp_db();
        let report = migrate_from_v0_1(&old_path, &new_conn).expect("migrate must not panic");

        assert_eq!(report.total_old_memories, 1);
        assert_eq!(report.entities_created, 1);
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);

        let key: String = new_conn
            .query_row(
                "SELECT key FROM entities WHERE id = ?1",
                params![evil_id],
                |r| r.get(0),
            )
            .unwrap();
        // Boundary walked back from 20 to 19: the é is dropped, not split.
        assert_eq!(key, format!("migrated-{}", "x".repeat(19)));

        let _ = std::fs::remove_file(&old_path);
    }

    #[test]
    fn gather_stats_returns_expected_shape() {
        let (conn, path) = temp_db();
        initialize_schema(&conn).expect("init schema");

        let now = now_ms();
        conn.execute(
            "INSERT INTO entities (id, category, key, body_json, created_at_unix_ms, last_accessed_unix_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params!["mem-1", "decision", "test-decision", "{}", now, now],
        )
        .unwrap();

        let stats = gather_stats(&conn, &path).unwrap();
        assert_eq!(stats.total_entities, 1);
        assert!(stats.db_file_size_bytes > 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn gather_stats_splits_active_and_archived_counts() {
        // #493: total_entities counted archived rows with no active-only
        // counterpart, so every user-visible number was inflated relative to
        // what list/count/recall (all archived = 0) actually return.
        let (conn, path) = temp_db();
        initialize_schema(&conn).expect("init schema");

        let now = now_ms();
        for (id, key, category, archived) in [
            ("mem-1", "k1", "decision", 0),
            ("mem-2", "k2", "decision", 0),
            ("mem-3", "k3", "decision", 1),
            ("mem-4", "k4", "insight", 1),
        ] {
            conn.execute(
                "INSERT INTO entities (id, category, key, body_json, archived,
                                       created_at_unix_ms, last_accessed_unix_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![id, category, key, "{}", archived, now, now],
            )
            .unwrap();
        }

        let stats = gather_stats(&conn, &path).unwrap();
        // Compatibility: the unsuffixed fields stay archived-inclusive.
        assert_eq!(stats.total_entities, 4);
        assert_eq!(stats.by_category["decision"], 3);
        assert_eq!(stats.by_category["insight"], 1);
        // New additive fields expose the view recall/list actually serve.
        assert_eq!(stats.active_entities, 2);
        assert_eq!(stats.archived_entities, 2);
        assert_eq!(stats.by_category_active["decision"], 2);
        assert!(stats.by_category_active.get("insight").is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fresh_db_has_artifact_tables_and_indexes() {
        let (conn, path) = temp_db();
        initialize_schema(&conn).expect("init schema");
        for table in ["artifacts", "artifact_bindings"] {
            let exists: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    params![table],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(exists, 1, "missing table {table}");
        }
        for index in [
            "idx_artifact_bindings_scope",
            "idx_artifact_bindings_derived_from",
        ] {
            let exists: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?1",
                    params![index],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(exists, 1, "missing index {index}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn v59_migration_adds_experience_projection_tables_and_indexes() {
        let (conn, path) = temp_db();
        initialize_schema(&conn).expect("initial schema");
        conn.execute_batch(
            "DROP TABLE experience_projection_events;
             DROP TABLE experience_projection_sources;
             DROP TABLE experience_projections;
             PRAGMA user_version = 58;",
        )
        .expect("reset fixture to v58");
        initialize_schema(&conn).expect("v58 to v59 migration");

        for table in [
            "experience_projections",
            "experience_projection_sources",
            "experience_projection_events",
        ] {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    params![table],
                    |row| row.get(0),
                )
                .expect("table lookup");
            assert_eq!(count, 1, "missing v59 table {table}");
        }
        for index in [
            "idx_experience_projections_source_state",
            "idx_experience_projection_sources_entity",
            "idx_experience_projection_events_scope",
        ] {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?1",
                    params![index],
                    |row| row.get(0),
                )
                .expect("index lookup");
            assert_eq!(count, 1, "missing v59 index {index}");
        }
        assert_eq!(
            conn.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            SCHEMA_VERSION
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fresh_db_has_rejected_value_tombstone_table_and_indexes() {
        // v26 (#849): digest-only scoped rejection tombstones.
        let (conn, path) = temp_db();
        initialize_schema(&conn).expect("init schema");
        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='rejected_value_tombstones'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(exists, 1, "missing rejected_value_tombstones table");
        for index in [
            "idx_rejected_tombstones_identity",
            "idx_rejected_tombstones_scope",
        ] {
            let idx: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?1",
                    params![index],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(idx, 1, "missing index {index}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn v01_migration_avoids_truncated_id_collisions() {
        let (source, source_path) = temp_db();
        source
            .execute_batch(
                "CREATE TABLE memories (
                    id TEXT PRIMARY KEY, content TEXT NOT NULL,
                    type TEXT DEFAULT 'insight', summary TEXT DEFAULT '',
                    relevance REAL DEFAULT 0.0, decay_score REAL DEFAULT 1.0,
                    retrieval_count INTEGER DEFAULT 0, layer TEXT DEFAULT 'working',
                    topic_path TEXT DEFAULT '', created_at_unix_ms INTEGER NOT NULL,
                    last_accessed_unix_ms INTEGER NOT NULL, workspace_hash TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]', links TEXT DEFAULT '[]', source TEXT DEFAULT 'legacy',
                    verified INTEGER DEFAULT 0
                );
                INSERT INTO memories
                    (id, content, created_at_unix_ms, last_accessed_unix_ms)
                VALUES
                    ('abcdefghijklmnopqrst-A', 'first legacy body', 1, 1),
                    ('abcdefghijklmnopqrst-B', 'second legacy body', 2, 2);",
            )
            .unwrap();
        drop(source);

        let (target, target_path) = temp_db();
        let report = migrate_from_v0_1(&source_path, &target).unwrap();
        assert_eq!(report.entities_created, 2);
        let rows: Vec<(String, String)> = target
            .prepare("SELECT id, key FROM entities ORDER BY id")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        assert_eq!(
            rows.len(),
            2,
            "truncated ids must not overwrite one another"
        );
        assert_ne!(
            rows[0].1, rows[1].1,
            "generated migration keys must be unique"
        );

        let _ = std::fs::remove_file(&source_path);
        let _ = std::fs::remove_file(&target_path);
    }

    #[test]
    fn v01_migration_preserves_workspace_column() {
        let (source, source_path) = temp_db();
        source
            .execute_batch(
                "CREATE TABLE memories (
                    id TEXT PRIMARY KEY, content TEXT NOT NULL,
                    type TEXT DEFAULT 'insight', summary TEXT DEFAULT '',
                    relevance REAL DEFAULT 0.0, decay_score REAL DEFAULT 1.0,
                    retrieval_count INTEGER DEFAULT 0, layer TEXT DEFAULT 'working',
                    topic_path TEXT DEFAULT '', created_at_unix_ms INTEGER NOT NULL,
                    last_accessed_unix_ms INTEGER NOT NULL, workspace_hash TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]', links TEXT DEFAULT '[]', source TEXT DEFAULT 'legacy',
                    verified INTEGER DEFAULT 0
                );
                INSERT INTO memories
                    (id, content, type, created_at_unix_ms, last_accessed_unix_ms, workspace_hash)
                VALUES ('workspace-memory', 'scoped legacy body', 'insight', 1, 1, 'ws-legacy');",
            )
            .unwrap();
        let (target, target_path) = temp_db();
        let report = migrate_from_v0_1(&source_path, &target).unwrap();
        assert_eq!(report.entities_created, 1, "legacy row must be imported");
        let workspace: String = target
            .query_row(
                "SELECT workspace_hash FROM entities WHERE id = 'workspace-memory'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(workspace, "ws-legacy");
        let _ = std::fs::remove_file(&source_path);
        let _ = std::fs::remove_file(&target_path);
    }
}
