//! #997: revocation credential-relative cutoff tests (concept borrowed from
//! RunAlphaLoop/verity, M1: "revocation with credential-relative cutoff").
//!
//! Contract under test:
//!   * workspace-scoped revocations subtract the principal from every grant
//!     set of manifests MINTED BEFORE the revocation — for the credential's
//!     full lifetime (it never grows the principal back).
//!   * manifests minted AFTER the revocation may carry the principal again
//!     (the operator deliberately re-granted — the cutoff is
//!     credential-relative, not wall-clock-absolute).
//!   * global (workspace '') revocations are the durable deprovisioned set:
//!     unconditional subtraction, regardless of mint time; a deprovisioned
//!     agent loses its manifest entirely (fail-closed over-hide).
//!   * reinstatement is explicit and durable; re-revocation after
//!     reinstatement appends a NEW revocation with its own cutoff.
//!   * revocation rows are durable-before-ack (committed before the journal
//!     write returns).
//!   * `include_revoked` (admin inspection) returns the RAW record —
//!     subtraction never mutates stored manifests.

use serde_json::json;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{now_ms, TestDatabase};
    use crate::models::{AuthorityManifestInput, MemoryLink};

    const WS: &str = "tenant-a";

    fn manifest_input(agent: &str, inbound: &[&str], approvers: &[&str]) -> AuthorityManifestInput {
        AuthorityManifestInput {
            agent_id: agent.to_string(),
            workspace_hash: WS.to_string(),
            allowed_capabilities: vec!["memory:read".to_string()],
            approval_required_capabilities: vec![],
            scope_anchors: vec!["urn:tenant-a:acme".to_string()],
            approver_principals: approvers.iter().map(|s| s.to_string()).collect(),
            allowed_inbound_principals: inbound.iter().map(|s| s.to_string()).collect(),
            permitted_external_ref_prefixes: vec![],
            max_parallel_actions: 1,
            mode: "enforce".to_string(),
            expires_at_unix_ms: None,
            capability_constraints_json: "{}".to_string(),
        }
    }

    fn seeded() -> TestDatabase {
        let db = TestDatabase::new("revocation-cutoff");
        db.agent_upsert("agent-owner", "owner", 0, "fleet-a")
            .unwrap();
        db.agent_upsert("agent-peer", "peer", 0, "fleet-a").unwrap();
        db.agent_upsert("agent-attacker", "attacker", 0, "fleet-b")
            .unwrap();
        db
    }

    fn manifest_with(
        db: &TestDatabase,
        agent: &str,
        inbound: &[&str],
        approvers: &[&str],
    ) -> crate::models::AuthorityManifest {
        db.authority_set(&manifest_input(agent, inbound, approvers), "operator")
            .expect("manifest set")
    }

    #[test]
    fn scoped_revocation_bites_manifests_minted_before() {
        let db = seeded();
        let m = manifest_with(
            &db,
            "agent-owner",
            &["agent-peer", "agent-attacker"],
            &["agent-peer"],
        );
        db.record_revocation("agent-attacker", WS, "compromised")
            .unwrap();
        let effective = db
            .authority_get("agent-owner", WS, false)
            .unwrap()
            .expect("manifest still active");
        assert!(
            effective.allowed_inbound_principals == vec!["agent-peer".to_string()],
            "revoked inbound principal must drop from the grant set: {:?}",
            effective.allowed_inbound_principals
        );
        assert_eq!(
            effective.approver_principals,
            vec!["agent-peer".to_string()]
        );
        assert!(
            m.revoked_at_unix_ms.is_none(),
            "subtraction must not mutate the stored record"
        );
    }

    #[test]
    fn scoped_revocation_spares_manifests_minted_after() {
        let db = seeded();
        // Revoke BEFORE the manifest is minted: the operator then re-granted
        // the principal deliberately. The old revocation must not bite the
        // newer credential.
        db.record_revocation("agent-attacker", WS, "incident a")
            .unwrap();
        let _m = manifest_with(&db, "agent-owner", &["agent-attacker"], &[]);
        let effective = db
            .authority_get("agent-owner", WS, false)
            .unwrap()
            .expect("manifest active");
        assert!(
            effective
                .allowed_inbound_principals
                .contains(&"agent-attacker".to_string()),
            "post-mint re-grant must survive: {:?}",
            effective.allowed_inbound_principals
        );
    }

    #[test]
    fn global_deprovision_drops_unconditionally() {
        let db = seeded();
        let _m = manifest_with(&db, "agent-owner", &["agent-attacker"], &[]);
        // GLOBAL revocation (workspace '') = durable deprovisioned set: no
        // credential, however new, may ever carry the principal again.
        db.record_revocation("agent-attacker", "", "decommissioned")
            .unwrap();
        let effective = db
            .authority_get("agent-owner", WS, false)
            .unwrap()
            .expect("manifest active");
        assert!(
            effective.allowed_inbound_principals.is_empty(),
            "deprovisioned principal must drop unconditionally: {:?}",
            effective.allowed_inbound_principals
        );
        // Even a manifest minted AFTER the deprovisioning cannot re-grant.
        let _new = manifest_with(&db, "agent-owner", &["agent-attacker"], &[]);
        let effective2 = db
            .authority_get("agent-owner", WS, false)
            .unwrap()
            .expect("manifest active");
        assert!(effective2.allowed_inbound_principals.is_empty());
    }

    #[test]
    fn deprovisioned_agent_loses_its_manifest() {
        let db = seeded();
        let _m = manifest_with(&db, "agent-owner", &[], &[]);
        db.record_revocation("agent-owner", "", "decommissioned")
            .unwrap();
        let got = db.authority_get("agent-owner", WS, false).unwrap();
        assert!(
            got.is_none(),
            "a deprovisioned agent's manifest must fail closed (over-hide)"
        );
    }

    #[test]
    fn reinstatement_restores_grants_and_is_durable() {
        let db = seeded();
        let _m = manifest_with(&db, "agent-owner", &["agent-attacker"], &[]);
        db.record_revocation("agent-attacker", WS, "suspected")
            .unwrap();
        db.reinstate_revocation("agent-attacker", WS).unwrap();
        let effective = db
            .authority_get("agent-owner", WS, false)
            .unwrap()
            .expect("manifest active");
        assert!(
            effective
                .allowed_inbound_principals
                .contains(&"agent-attacker".to_string()),
            "reinstatement must restore the grant"
        );
    }

    #[test]
    fn re_revocation_after_reinstatement_appends_a_new_cutoff() {
        let db = seeded();
        let _m = manifest_with(&db, "agent-owner", &["agent-attacker"], &[]);
        db.record_revocation("agent-attacker", WS, "first").unwrap();
        db.reinstate_revocation("agent-attacker", WS).unwrap();
        // Re-revoke after reinstatement: a NEW row with its own `at` — and it
        // bites again (the manifest predates it).
        db.record_revocation("agent-attacker", WS, "second")
            .unwrap();
        let effective = db
            .authority_get("agent-owner", WS, false)
            .unwrap()
            .expect("manifest active");
        assert!(
            effective.allowed_inbound_principals.is_empty(),
            "re-revocation must bite again"
        );
        let conn = db.conn().unwrap();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM revocations WHERE principal='agent-attacker'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 2, "append-only ledger must hold both revocations");
    }

    #[test]
    fn revocation_rows_commit_before_ack() {
        let db = seeded();
        db.record_revocation("agent-attacker", WS, "durable-before-ack")
            .unwrap();
        // No further writes: the row is visible to a fresh connection NOW.
        let conn = db.conn().unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM revocations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn include_revoked_returns_the_raw_record() {
        let db = seeded();
        let m = manifest_with(&db, "agent-owner", &["agent-attacker"], &["agent-peer"]);
        db.record_revocation("agent-attacker", WS, "raw-check")
            .unwrap();
        let raw = db
            .authority_get("agent-owner", WS, true)
            .unwrap()
            .expect("raw manifest");
        assert!(
            raw.allowed_inbound_principals
                .contains(&"agent-attacker".to_string()),
            "admin inspection must show the stored record, not the subtraction"
        );
        assert_eq!(raw.id, m.id);
    }

    #[test]
    fn revocation_subtracts_all_grant_fields() {
        let db = seeded();
        let _m = manifest_with(&db, "agent-owner", &["agent-attacker"], &["agent-attacker"]);
        // Put the principal into scope_anchors via a second manifest input.
        let mut inp = manifest_input("agent-owner", &["agent-attacker"], &["agent-attacker"]);
        inp.scope_anchors = vec![
            "urn:tenant-a:acme".to_string(),
            "agent-attacker".to_string(),
        ];
        let _m2 = db.authority_set(&inp, "operator").unwrap();
        db.record_revocation("agent-attacker", WS, "all-fields")
            .unwrap();
        let effective = db
            .authority_get("agent-owner", WS, false)
            .unwrap()
            .expect("manifest active");
        assert!(effective.allowed_inbound_principals.is_empty());
        assert!(effective.approver_principals.is_empty());
        assert!(
            !effective
                .scope_anchors
                .iter()
                .any(|a| a == "agent-attacker"),
            "scope anchors must drop the revoked principal: {:?}",
            effective.scope_anchors
        );
    }

    #[test]
    fn subtraction_is_idempotent_across_repeated_loads() {
        let db = seeded();
        let _m = manifest_with(&db, "agent-owner", &["agent-attacker"], &[]);
        db.record_revocation("agent-attacker", WS, "x").unwrap();
        for _ in 0..3 {
            let e = db
                .authority_get("agent-owner", WS, false)
                .unwrap()
                .expect("manifest active");
            assert!(e.allowed_inbound_principals.is_empty());
        }
    }
}
