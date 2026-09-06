//! #999: derived-visibility intersection tests (concept borrowed from
//! RunAlphaLoop/verity, M1). A write that declares lineage via links with
//! relationship `derived_from` must not become more open than its cited
//! inputs; unreadable or missing inputs refuse the whole write fail-closed.

use serde_json::json;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::TestDatabase;

    fn entity(
        category: &str,
        key: &str,
        body: &str,
        ws: &str,
        agent: &str,
        visibility: &str,
    ) -> crate::models::Entity {
        let raw = uuid::Uuid::new_v4().simple().to_string();
        serde_json::from_value(json!({
            "id": format!("mem-{}", &raw[..12]),
            "category": category,
            "key": key,
            "body_json": body,
            "workspace_hash": ws,
            "agent_id": agent,
            "visibility": visibility,
            "created_at_unix_ms": 0,
            "last_accessed_unix_ms": 0,
        }))
        .expect("entity deserializes")
    }

    fn derived(
        category: &str,
        key: &str,
        body: &str,
        ws: &str,
        agent: &str,
        visibility: &str,
        targets: &[String],
    ) -> crate::models::Entity {
        let mut e = entity(category, key, body, ws, agent, visibility);
        e.links = targets
            .iter()
            .map(|t| crate::models::MemoryLink {
                target_id: t.clone(),
                relationship: "derived_from".to_string(),
                weight: 0.5,
                source: Some("test".to_string()),
                kind: None,
                asserted_at_unix_ms: None,
            })
            .collect();
        e
    }

    fn stored_visibility(db: &TestDatabase, category: &str, key: &str) -> String {
        let e = db
            .get_entity(category, key)
            .expect("lookup ok")
            .expect("stored entity exists");
        e.visibility
    }

    #[test]
    fn derived_from_private_input_cannot_become_workspace_visible() {
        let db = TestDatabase::new("derived-vis");
        db.agent_upsert("agent-a", "owner", 0, "fleet-a").unwrap();
        // The classic Verity failure: summarize a private source into shared
        // memory. The derived row must inherit the input's privacy.
        let (source_id, _) = db
            .remember(&entity(
                "pricing",
                "acme-renewal",
                "ACME renewal is $61k floor $58k",
                "tenant-a",
                "agent-a",
                "private",
            ))
            .unwrap();
        let (derived_id, _) = db
            .remember(&derived(
                "summary",
                "renewal-benchmarks",
                "renewal pricing benchmarks: ACME $61k",
                "tenant-a",
                "agent-a",
                "workspace", // the writer DECLARES open visibility
                &[source_id],
            ))
            .unwrap();
        let stored = db
            .get_entity_by_id_public(&derived_id)
            .unwrap()
            .expect("derived row exists");
        assert_eq!(
            stored.visibility, "private",
            "a summary of a private source must be private"
        );
    }

    #[test]
    fn intersection_takes_most_restrictive_input() {
        let db = TestDatabase::new("derived-vis");
        db.agent_upsert("agent-a", "owner", 0, "fleet-a").unwrap();
        let (workspace_src, _) = db
            .remember(&entity(
                "notes",
                "roadmap",
                "roadmap body",
                "tenant-a",
                "agent-a",
                "workspace",
            ))
            .unwrap();
        let (public_src, _) = db
            .remember(&entity(
                "notes", "faq", "faq body", "tenant-a", "agent-a", "public",
            ))
            .unwrap();
        let (derived_id, _) = db
            .remember(&derived(
                "summary",
                "roadmap-plus-faq",
                "combined",
                "tenant-a",
                "agent-a",
                "public",
                &[workspace_src, public_src],
            ))
            .unwrap();
        let stored = db
            .get_entity_by_id_public(&derived_id)
            .unwrap()
            .expect("derived row exists");
        assert_eq!(
            stored.visibility, "workspace",
            "public ∩ workspace = workspace (most restrictive input wins)"
        );
    }

    #[test]
    fn declared_stricter_visibility_is_kept() {
        let db = TestDatabase::new("derived-vis");
        db.agent_upsert("agent-a", "owner", 0, "fleet-a").unwrap();
        let (src, _) = db
            .remember(&entity(
                "notes",
                "src",
                "body",
                "tenant-a",
                "agent-a",
                "workspace",
            ))
            .unwrap();
        let (derived_id, _) = db
            .remember(&derived(
                "summary",
                "private-summary",
                "writer chose private",
                "tenant-a",
                "agent-a",
                "private",
                &[src],
            ))
            .unwrap();
        let stored = db
            .get_entity_by_id_public(&derived_id)
            .unwrap()
            .expect("derived row exists");
        assert_eq!(
            stored.visibility, "private",
            "a stricter declared visibility must not be widened by the intersection"
        );
    }

    #[test]
    fn unreadable_input_refuses_the_whole_write() {
        let db = TestDatabase::new("derived-vis");
        db.agent_upsert("agent-a", "victim", 0, "fleet-a").unwrap();
        db.agent_upsert("agent-b", "attacker", 0, "fleet-b")
            .unwrap();
        let (victim_src, _) = db
            .remember(&entity(
                "pricing",
                "acme-renewal",
                "ACME $61k",
                "tenant-a",
                "agent-a",
                "private",
            ))
            .unwrap();
        // agent-b cites a row it cannot read: refuse, persist nothing.
        let err = db
            .remember(&derived(
                "summary",
                "stolen-summary",
                "ACME pays $61k",
                "tenant-b",
                "agent-b",
                "workspace",
                &[victim_src],
            ))
            .expect_err("write must be refused");
        assert!(
            err.to_string().contains("not visible"),
            "error must name the visibility refusal: {err}"
        );
        let stored = db.get_entity("summary", "stolen-summary").unwrap();
        assert!(stored.is_none(), "refused write must persist nothing");
    }

    #[test]
    fn missing_lineage_target_refuses_the_write() {
        let db = TestDatabase::new("derived-vis");
        db.agent_upsert("agent-a", "owner", 0, "fleet-a").unwrap();
        let err = db
            .remember(&derived(
                "summary",
                "dangling",
                "cites nothing",
                "tenant-a",
                "agent-a",
                "workspace",
                &["mem-does-not-exist".to_string()],
            ))
            .expect_err("dangling lineage must be refused");
        assert!(
            err.to_string().contains("not found"),
            "error must name the missing target: {err}"
        );
    }

    #[test]
    fn non_lineage_links_do_not_trigger_intersection() {
        let db = TestDatabase::new("derived-vis");
        db.agent_upsert("agent-a", "owner", 0, "fleet-a").unwrap();
        let (other, _) = db
            .remember(&entity(
                "notes",
                "other",
                "other row body",
                "tenant-a",
                "agent-a",
                "private",
            ))
            .unwrap();
        // A related/similar link (NOT derived_from) carries no inheritance
        // semantics — visibility stays exactly as declared. (Distinct bodies:
        // identical (workspace, body) rows would collide in the dedup path.)
        let mut e = entity(
            "notes",
            "related-row",
            "related row body",
            "tenant-a",
            "agent-a",
            "workspace",
        );
        e.links = vec![crate::models::MemoryLink {
            target_id: other,
            relationship: "related".to_string(),
            weight: 0.5,
            source: Some("test".to_string()),
            kind: None,
            asserted_at_unix_ms: None,
        }];
        let (id, _) = db.remember(&e).unwrap();
        let stored = db.get_entity_by_id_public(&id).unwrap().unwrap();
        assert_eq!(stored.visibility, "workspace");
    }

    #[test]
    fn legacy_write_without_lineage_keeps_declared_visibility() {
        let db = TestDatabase::new("derived-vis");
        db.agent_upsert("agent-a", "owner", 0, "fleet-a").unwrap();
        let (id, _) = db
            .remember(&entity(
                "notes", "legacy", "body", "tenant-a", "agent-a", "public",
            ))
            .unwrap();
        let stored = db.get_entity_by_id_public(&id).unwrap().unwrap();
        assert_eq!(stored.visibility, "public");
    }
}
