#![allow(clippy::too_many_arguments)]

#[path = "../src/context_transform.rs"]
mod context_transform;
#[path = "../src/source_chain.rs"]
mod source_chain;

use context_transform::{
    digest_messages, replay_membership, transform_context, BoundedReference, ContextMessage,
    ContextTransformRequest, ReplayMembership, ReplayPlan, TransformStage, TransformerDescriptor,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn sha(seed: &str) -> String {
    digest_messages(&[ContextMessage::new(
        "digest-seed",
        0,
        "assistant_prose",
        json!({"role": "assistant", "content": seed}),
    )])
    .expect("digest fixture")
}

fn legacy_digest<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("legacy digest fixture");
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Serialize)]
struct LegacyReplayMembership<'a> {
    source_id: &'a str,
    input_order: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_order: Option<u32>,
    content_class: &'a str,
    input_digest: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_digest: Option<&'a str>,
    disposition: &'a str,
}

fn stage(trust: &str) -> TransformStage {
    TransformStage::new(
        "fixture-distill",
        "0.3.7",
        true,
        trust,
        Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string()),
    )
}

fn request(
    messages: Vec<ContextMessage>,
    policy: &str,
    trust: &str,
    with_references: bool,
) -> ContextTransformRequest {
    let mut request = ContextTransformRequest::new(
        "openai",
        "openai-chat/v1",
        TransformerDescriptor::new("fixture-transformer", "1.0.0"),
        vec![stage(trust)],
        policy,
        messages,
    )
    .with_input_tokens(Some(120));
    if with_references {
        request = request
            .with_original_ref(BoundedReference::new(
                "vault-retrieval",
                "fixture-original",
                sha("original"),
            ))
            .with_replay_envelope_ref(BoundedReference::new(
                "retrieval-replay",
                "fixture-replay",
                sha("replay"),
            ));
    }
    request
}

fn message(id: &str, order: u32, class: &str, role: &str, content: Value) -> ContextMessage {
    ContextMessage::new(id, order, class, json!({"role": role, "content": content}))
}

fn tool_pair() -> Vec<ContextMessage> {
    vec![
        ContextMessage::new(
            "tool-use-1",
            0,
            "tool_use",
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-critical",
                    "type": "function",
                    "function": {
                        "name": "inspect_records",
                        "arguments": "{\"limit\": 50}"
                    }
                }]
            }),
        ),
        ContextMessage::new(
            "tool-result-1",
            1,
            "tool_result",
            json!({
                "role": "tool",
                "tool_call_id": "call-critical",
                "content": "{\"records\":[{\"id\":\"head\"}],\"critical_tail\":\"MUST_KEEP_TOOL_TAIL\"}"
            }),
        ),
    ]
}

#[test]
fn untrusted_distillation_cannot_remove_user_constraints() {
    let input = vec![
        message("system-1", 0, "system", "system", "policy marker".into()),
        message(
            "requirements-1",
            1,
            "user_constraint",
            "user",
            "MUST_KEEP_REQUIREMENT_A; MUST_KEEP_REQUIREMENT_B".into(),
        ),
        message(
            "assistant-1",
            2,
            "assistant_prose",
            "assistant",
            "explanation".into(),
        ),
    ];
    let mut proposed = input.clone();
    proposed[1].message["content"] = json!("MUST_KEEP_REQUIREMENT_A");

    let decision = transform_context(
        &request(input.clone(), "reversible", "untrusted", false),
        proposed,
        Some(30),
    )
    .expect("contract evaluation");

    assert_eq!(decision.receipt.outcome, "rejected");
    assert_eq!(decision.output_messages, input);
    assert!(decision
        .receipt
        .reason_code
        .as_deref()
        .unwrap_or_default()
        .contains("protected"));
    assert!(decision.output_messages[1].message["content"]
        .as_str()
        .unwrap()
        .contains("MUST_KEEP_REQUIREMENT_B"));
}

#[test]
fn wire_shape_pairing_does_not_make_a_lossy_tool_result_reversible() {
    let input = tool_pair();
    let mut proposed = input.clone();
    proposed[1].message["content"] = json!("{\"records\":[{\"id\":\"head\"}]} ");

    let decision = transform_context(
        &request(input.clone(), "reversible", "trusted", false),
        proposed,
        Some(18),
    )
    .expect("contract evaluation");

    assert_eq!(decision.receipt.outcome, "rejected");
    assert_eq!(decision.output_messages, input);
    assert_eq!(decision.receipt.actual_lossiness, "none");
    assert!(decision
        .receipt
        .reason_code
        .as_deref()
        .unwrap_or_default()
        .contains("recover"));
}

#[test]
fn explicitly_opted_in_loss_is_degraded_and_marked_lossy() {
    let input = tool_pair();
    let mut proposed = input.clone();
    proposed[1].message["content"] = json!("{\"records\": []}");

    let decision = transform_context(
        &request(input, "lossy_opt_in", "trusted", false),
        proposed.clone(),
        Some(10),
    )
    .expect("contract evaluation");

    assert_eq!(decision.receipt.outcome, "degraded");
    assert_eq!(decision.receipt.actual_lossiness, "lossy");
    assert_eq!(decision.output_messages, proposed);
    assert!(!decision.receipt.original_retained);
    assert!(decision.receipt.replay.is_some());
}

#[test]
fn reversible_receipt_seals_exact_digests_counts_and_stage_configuration() {
    let input = vec![
        message(
            "assistant-1",
            0,
            "assistant_prose",
            "assistant",
            "long explanation".into(),
        ),
        message(
            "fenced-1",
            1,
            "fenced",
            "user",
            "```rust\nfn critical_tail() {}\n```".into(),
        ),
    ];
    let mut proposed = input.clone();
    proposed[0].message["content"] = json!("short explanation");
    let request = request(input.clone(), "reversible", "trusted", true);

    let decision =
        transform_context(&request, proposed.clone(), Some(32)).expect("contract evaluation");
    let receipt = decision.receipt;

    assert_eq!(receipt.outcome, "transformed");
    assert_eq!(receipt.actual_lossiness, "reversible");
    assert_eq!(receipt.input_digest, digest_messages(&input).unwrap());
    assert_eq!(receipt.output_digest, digest_messages(&proposed).unwrap());
    assert_eq!(receipt.token_counts.input, Some(120));
    assert_eq!(receipt.token_counts.output, Some(32));
    assert_eq!(receipt.stages[0].name, "fixture-distill");
    assert_eq!(
        receipt.stages[0].config_digest.as_deref(),
        Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    );
    assert!(receipt.original_retained);
    assert!(receipt.replay.is_some());
    assert_eq!(receipt.digest().unwrap().len(), 64);
    assert!(receipt.validate().is_ok());
}

#[test]
fn legacy_replay_envelope_without_source_chain_still_replays() {
    let input = message(
        "legacy-1",
        0,
        "assistant_prose",
        "assistant",
        "legacy content".into(),
    );
    let input_digest = legacy_digest(&(input.content_class.as_str(), &input.message));
    let output_digest = input_digest.clone();
    let legacy_member = LegacyReplayMembership {
        source_id: "legacy-1",
        input_order: 0,
        output_order: Some(0),
        content_class: "assistant_prose",
        input_digest: &input_digest,
        output_digest: Some(&output_digest),
        disposition: "retained",
    };
    let fingerprint = legacy_digest(&vec![legacy_member]);
    let plan = ReplayPlan {
        envelope_ref: None,
        original_ref: None,
        membership: vec![ReplayMembership {
            source_id: "legacy-1".to_string(),
            input_order: 0,
            output_order: Some(0),
            content_class: "assistant_prose".to_string(),
            input_digest,
            output_digest: Some(output_digest),
            disposition: "retained".to_string(),
            source_chain: Default::default(),
        }],
        fingerprint,
    };

    assert!(plan.validate().is_ok());
    let replayed = replay_membership(&plan, &[input]).expect("legacy replay");
    assert_eq!(replayed.ordered_source_ids, vec!["legacy-1"]);
}

#[test]
fn known_source_chain_rejects_body_only_legacy_digest() {
    let identity = source_chain::SourceChainIdentity::from_body(&json!({
        "source_chain": {"schema_version": 1, "chain_id": "chain-known"}
    }))
    .unwrap();
    let input = message(
        "known-legacy-1",
        0,
        "assistant_prose",
        "assistant",
        "legacy body".into(),
    )
    .with_source_chain(identity.clone());
    let input_digest = legacy_digest(&(input.content_class.as_str(), &input.message));
    let member = ReplayMembership {
        source_id: input.id.clone(),
        input_order: 0,
        output_order: Some(0),
        content_class: input.content_class.clone(),
        input_digest,
        output_digest: Some(sha("known-output")),
        disposition: "retained".to_string(),
        source_chain: identity,
    };
    let fingerprint = legacy_digest(&vec![member.clone()]);
    let plan = ReplayPlan {
        envelope_ref: None,
        original_ref: None,
        membership: vec![member],
        fingerprint,
    };

    assert!(plan.validate().is_ok());
    assert!(replay_membership(&plan, &[input]).is_err());
}

#[test]
fn public_receipts_expose_only_source_chain_commitments() {
    let identity = source_chain::SourceChainIdentity::from_body(&json!({
        "source_chain": {
            "schema_version": 1,
            "source_group_id": "group-a",
            "chain_id": "chain-a"
        }
    }))
    .unwrap();
    let input = vec![message(
        "receipt-lineage-1",
        0,
        "assistant_prose",
        "assistant",
        "original".into(),
    )
    .with_source_chain(identity)];
    let mut proposed = input.clone();
    proposed[0].message["content"] = json!("transformed");
    let decision = transform_context(
        &request(input, "reversible", "trusted", true),
        proposed,
        Some(32),
    )
    .expect("receipt contract evaluation");
    let replay = decision.receipt.replay.expect("replay receipt");
    let public = serde_json::to_value(&replay).expect("public replay receipt");
    let source_chain = &public["membership"][0]["source_chain"];
    assert_eq!(source_chain["status"], json!("known"));
    assert!(source_chain["commitment_sha256"].as_str().is_some());
    for field in [
        "schema_version",
        "source_group_id",
        "episode_id",
        "experience_id",
        "chain_id",
        "thread_id",
        "subject_ids",
        "parent_id",
        "sequence",
        "valid_from_unix_ms",
        "valid_to_unix_ms",
    ] {
        assert!(
            source_chain.get(field).is_none(),
            "unexpected public field: {field}"
        );
    }
    let unknown = serde_json::to_value(source_chain::SourceChainIdentity::unknown())
        .expect("unknown source-chain projection");
    assert_eq!(unknown["status"], json!("unknown"));
    assert!(unknown.get("commitment_sha256").is_none());
    let unknown_membership = ReplayMembership {
        source_id: "unknown-source".to_string(),
        input_order: 0,
        output_order: None,
        content_class: "assistant_prose".to_string(),
        input_digest: sha("unknown"),
        output_digest: None,
        disposition: "omitted".to_string(),
        source_chain: source_chain::SourceChainIdentity::unknown(),
    };
    let unknown_membership_public =
        serde_json::to_value(unknown_membership).expect("unknown membership projection");
    assert_eq!(
        unknown_membership_public["source_chain"]["status"],
        json!("unknown")
    );
    assert!(unknown_membership_public["source_chain"]
        .get("commitment_sha256")
        .is_none());
}

#[test]
fn source_chain_change_is_not_treated_as_retained_content() {
    let first = source_chain::SourceChainIdentity::from_body(&json!({
        "source_chain": {"schema_version": 1, "chain_id": "chain-a"}
    }))
    .unwrap();
    let second = source_chain::SourceChainIdentity::from_body(&json!({
        "source_chain": {"schema_version": 1, "chain_id": "chain-b"}
    }))
    .unwrap();
    let input = vec![message(
        "lineage-1",
        0,
        "assistant_prose",
        "assistant",
        "unchanged content".into(),
    )
    .with_source_chain(first)];
    let proposed = vec![message(
        "lineage-1",
        0,
        "assistant_prose",
        "assistant",
        "unchanged content".into(),
    )
    .with_source_chain(second)];
    let decision = transform_context(
        &request(input, "reversible", "trusted", true),
        proposed,
        Some(32),
    )
    .expect("lineage change remains auditable");
    assert_eq!(decision.receipt.outcome, "transformed");
    assert!(decision
        .receipt
        .changed_content_classes
        .contains(&"assistant_prose".to_string()));
}

#[test]
fn replay_plan_preserves_membership_order_and_original_locator_without_bodies() {
    let input = vec![
        message(
            "user-1",
            0,
            "user_constraint",
            "user",
            "MUST_KEEP_REQUIREMENT_A".into(),
        ),
        message(
            "assistant-1",
            1,
            "assistant_prose",
            "assistant",
            "discardable prose".into(),
        ),
        message(
            "user-2",
            2,
            "assistant_prose",
            "user",
            "opaque content".into(),
        ),
    ];
    let mut retained_tail = input[2].clone();
    retained_tail.order = 1;
    let proposed = vec![input[0].clone(), retained_tail];
    let decision = transform_context(
        &request(input, "reversible", "trusted", true),
        proposed,
        Some(70),
    )
    .expect("contract evaluation");

    assert_eq!(decision.receipt.outcome, "transformed");
    let replay = decision.receipt.replay.as_ref().expect("replay plan");
    assert_eq!(replay.membership.len(), 3);
    assert_eq!(replay.membership[0].source_id, "user-1");
    assert_eq!(replay.membership[0].output_order, Some(0));
    assert_eq!(replay.membership[1].source_id, "assistant-1");
    assert_eq!(replay.membership[1].output_order, None);
    assert_eq!(replay.membership[1].disposition, "omitted");
    assert_eq!(replay.membership[2].source_id, "user-2");
    assert_eq!(replay.membership[2].output_order, Some(1));
    assert_eq!(replay.original_ref.as_ref().unwrap().id, "fixture-original");
    assert_eq!(replay.envelope_ref.as_ref().unwrap().id, "fixture-replay");
    assert_eq!(replay.fingerprint.len(), 64);

    let serialized = serde_json::to_string(&decision.receipt).unwrap();
    assert!(!serialized.contains("MUST_KEEP_REQUIREMENT_A"));
    assert!(!serialized.contains("discardable prose"));
    assert!(!serialized.contains("opaque content"));
}

#[test]
fn provider_adapter_replays_membership_order_and_locates_original_by_digest() {
    let input = vec![
        message(
            "user-1",
            0,
            "user_constraint",
            "user",
            "MUST_KEEP_REQUIREMENT_A".into(),
        ),
        message(
            "assistant-1",
            1,
            "assistant_prose",
            "assistant",
            "discardable prose".into(),
        ),
        message(
            "user-2",
            2,
            "assistant_prose",
            "user",
            "opaque content".into(),
        ),
    ];
    let mut retained_tail = input[2].clone();
    retained_tail.order = 1;
    let decision = transform_context(
        &request(input.clone(), "reversible", "trusted", true),
        vec![input[0].clone(), retained_tail],
        Some(70),
    )
    .expect("contract evaluation");
    let replay = decision.receipt.replay.as_ref().expect("replay plan");

    let replayed = context_transform::replay_membership(replay, &input).expect("replay membership");
    assert_eq!(replayed.ordered_source_ids, vec!["user-1", "user-2"]);
    assert_eq!(replayed.omitted_source_ids, vec!["assistant-1"]);
    assert_eq!(
        replayed.original_ref.as_ref().unwrap().id,
        "fixture-original"
    );
    assert_eq!(replayed.replay_fingerprint, replay.fingerprint);
}

#[test]
fn replay_preserves_source_chain_identity_commitments() {
    let identity = source_chain::SourceChainIdentity::from_body(&json!({
        "source_chain": {
            "schema_version": 1,
            "experience_id": "experience-1",
            "chain_id": "chain-a",
            "thread_id": "thread-1",
            "subject_ids": ["subject-a"],
            "sequence": 4
        }
    }))
    .expect("source-chain identity");
    let input = vec![message(
        "assistant-1",
        0,
        "assistant_prose",
        "assistant",
        "keep this".into(),
    )
    .with_source_chain(identity.clone())];
    let mut proposed = input.clone();
    proposed[0].message["content"] = json!("changed");
    let decision = transform_context(
        &request(input.clone(), "reversible", "trusted", true),
        proposed,
        Some(32),
    )
    .expect("contract evaluation");
    let replay = decision.receipt.replay.as_ref().expect("replay plan");
    assert_eq!(replay.membership[0].source_chain, identity);
    let replayed = context_transform::replay_membership(replay, &input).expect("replay");
    assert_eq!(replayed.ordered_source_ids, vec!["assistant-1"]);
    assert_eq!(
        replayed.source_chain_commitments,
        vec![Some(identity.commitment().to_string())]
    );
}

#[test]
fn replay_does_not_emit_unknown_source_chain_commitment() {
    let input = vec![message(
        "legacy-1",
        0,
        "assistant_prose",
        "assistant",
        "legacy content".into(),
    )];
    let mut proposed = input.clone();
    proposed[0].message["content"] = json!("changed legacy content");
    let decision = transform_context(
        &request(input.clone(), "reversible", "trusted", true),
        proposed,
        Some(32),
    )
    .expect("contract evaluation");
    let replay = decision.receipt.replay.as_ref().expect("replay plan");
    let replayed = context_transform::replay_membership(replay, &input).expect("replay");
    let serialized = serde_json::to_value(replayed).expect("serialized replay result");
    assert!(
        serialized["source_chain_commitments"][0].is_null(),
        "unknown source-chain identities must not expose commitments: {serialized}"
    );
}

#[test]
fn unknown_source_chain_is_not_treated_as_compatible() {
    let unknown = source_chain::SourceChainIdentity::unknown();
    assert!(!unknown.compatible_with(&unknown));
}

#[test]
fn unknown_content_and_unsupported_provider_format_default_to_passthrough() {
    let input = vec![message(
        "opaque-1",
        0,
        "unknown",
        "user",
        "original opaque".into(),
    )];
    let mut proposed = input.clone();
    proposed[0].message["content"] = json!("heuristically shortened");
    let decision = transform_context(
        &request(input.clone(), "lossy_opt_in", "trusted", false),
        proposed,
        Some(2),
    )
    .expect("contract evaluation");
    assert_eq!(decision.receipt.outcome, "passthrough");
    assert_eq!(decision.output_messages, input);

    let input = vec![message(
        "unsupported-1",
        0,
        "assistant_prose",
        "assistant",
        "original".into(),
    )];
    let mut unsupported = request(input.clone(), "lossy_opt_in", "trusted", false);
    unsupported.request_format = "future-provider-shape/v9".to_string();
    let mut proposed = input.clone();
    proposed[0].message["content"] = json!("changed");
    let decision = transform_context(&unsupported, proposed, Some(1)).expect("contract evaluation");
    assert_eq!(decision.receipt.outcome, "passthrough");
    assert_eq!(decision.output_messages, input);
}

#[test]
fn malformed_tool_pair_is_rejected_separately_from_content_recall() {
    let input = tool_pair();
    let proposed = vec![input[0].clone()];
    let decision = transform_context(
        &request(input.clone(), "lossy_opt_in", "trusted", false),
        proposed,
        Some(12),
    )
    .expect("contract evaluation");

    assert_eq!(decision.receipt.outcome, "rejected");
    assert_eq!(decision.output_messages, input);
    assert!(decision.receipt.provider_shape.pairing_checked);
    assert!(!decision.receipt.provider_shape.proposed_output_valid);
    assert!(decision
        .receipt
        .reason_code
        .as_deref()
        .unwrap_or_default()
        .contains("provider_shape"));
}

#[test]
fn tampering_receipt_digests_or_admitting_loss_without_recovery_fails_closed() {
    let input = vec![message(
        "assistant-1",
        0,
        "assistant_prose",
        "assistant",
        "long".into(),
    )];
    let mut proposed = input.clone();
    proposed[0].message["content"] = json!("short");
    let mut receipt = transform_context(
        &request(input, "reversible", "trusted", true),
        proposed,
        Some(1),
    )
    .unwrap()
    .receipt;
    receipt.input_digest =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string();
    assert!(receipt.validate().is_err());

    let input = tool_pair();
    let mut output = input.clone();
    output[1].message["content"] = json!("truncated");
    let mut no_recovery = request(input, "reversible", "trusted", false);
    no_recovery.original_retained = true;
    let error = transform_context(&no_recovery, output, Some(1)).unwrap_err();
    assert!(error.contains("original_retained"));
}

#[test]
fn synthetic_fixture_covers_required_roles_tails_pairs_and_multimodal_shape() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/context_transformer_v1.json"))
        .expect("valid transformer fixture");
    assert_eq!(
        fixture["contract_version"],
        "perseus-vault-context-transformer/v1"
    );
    let messages = fixture["messages"].as_array().expect("messages");
    for class in [
        "system",
        "keystone_policy",
        "authority",
        "user_constraint",
        "assistant_prose",
        "tool_use",
        "tool_result",
        "source_code",
        "fenced",
        "multimodal",
    ] {
        assert!(
            messages.iter().any(|m| m["content_class"] == class),
            "missing {class}"
        );
    }
    let requirements = messages
        .iter()
        .find(|m| m["content_class"] == "user_constraint")
        .unwrap();
    assert!(requirements["message"]["content"]
        .as_str()
        .unwrap()
        .contains("MUST_KEEP_REQUIREMENT_B"));
    let tool_result = messages
        .iter()
        .find(|m| m["content_class"] == "tool_result")
        .unwrap();
    assert!(tool_result["message"]["content"]
        .as_str()
        .unwrap()
        .contains("MUST_KEEP_TOOL_TAIL"));
    let code = messages
        .iter()
        .find(|m| m["content_class"] == "source_code")
        .unwrap();
    assert!(code["message"]["content"]
        .as_str()
        .unwrap()
        .contains("critical_tail_marker"));
    let fenced = messages
        .iter()
        .find(|m| m["content_class"] == "fenced")
        .unwrap();
    assert!(fenced["message"]["content"]
        .as_str()
        .unwrap()
        .contains("MUST_KEEP_FENCED_TAIL"));
    let multimodal = messages
        .iter()
        .find(|m| m["content_class"] == "multimodal")
        .unwrap();
    assert!(multimodal["message"]["content"].is_array());
    assert!(fixture["cases"].as_array().unwrap().len() >= 6);
}
