use std::collections::BTreeMap;

use serde::Serialize;

use crate::models::ArtifactAnchor;

pub const DIGEST_VERSION: &str = "evidence-log-digest-v1";
const PROTECTED: [&str; 10] = [
    "error",
    "warn",
    "warning",
    "exception",
    "fatal",
    "panic",
    "denied",
    "refused",
    "timeout",
    "assertion", // traceback is covered below
];

#[derive(Debug, Clone, Serialize)]
pub struct DigestSection {
    pub kind: String,
    pub template: String,
    pub count: usize,
    pub first: ArtifactAnchor,
    pub last: ArtifactAnchor,
    pub omitted_occurrences: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvidenceLogDigest {
    pub format: &'static str,
    pub source_sha256: String,
    pub config_version: &'static str,
    pub input_line_count: usize,
    pub omitted_line_count: usize,
    pub protected_line_count: usize,
    pub sections: Vec<DigestSection>,
    pub protected_lines: Vec<(String, ArtifactAnchor)>,
    pub retrieval: &'static str,
}

fn protected(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    l.contains("traceback") || PROTECTED.iter().any(|word| l.contains(word))
}

/// Normalize deterministic high-cardinality tokens only; this is navigation,
/// not evidence. Every result retains first/last anchors to original bytes.
fn template(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut token = String::new();
    let flush = |out: &mut String, token: &mut String| {
        if token.chars().all(|c| c.is_ascii_digit())
            || token.len() >= 16 && token.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
        {
            out.push_str("<value>");
        } else {
            out.push_str(token);
        }
        token.clear();
    };
    for c in line.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == ':' {
            token.push(c);
        } else {
            flush(&mut out, &mut token);
            out.push(c);
        }
    }
    flush(&mut out, &mut token);
    out
}

pub fn digest(
    sha256: &str,
    bytes: &[u8],
    anchor: impl Fn(usize, usize) -> ArtifactAnchor,
) -> Result<EvidenceLogDigest, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| "log digest requires a UTF-8 text artifact".to_string())?;
    let mut groups: BTreeMap<String, (usize, ArtifactAnchor, ArtifactAnchor)> = BTreeMap::new();
    let mut protected_lines = Vec::new();
    let mut start = 0usize;
    let mut total = 0usize;
    for raw in text.split_inclusive('\n') {
        let end = start + raw.len();
        let line = raw.trim_end_matches('\n').trim_end_matches('\r');
        total += 1;
        let a = anchor(start, end);
        if protected(line) {
            protected_lines.push((line.to_string(), a));
        } else {
            let key = template(line);
            groups
                .entry(key)
                .and_modify(|v| {
                    v.0 += 1;
                    v.2 = a.clone();
                })
                .or_insert((1, a.clone(), a));
        }
        start = end;
    }
    if text.is_empty() {
        total = 0;
    }
    let sections = groups
        .into_iter()
        .map(|(template, (count, first, last))| DigestSection {
            kind: "collapsed_template".to_string(),
            template,
            count,
            first,
            last,
            omitted_occurrences: count.saturating_sub(2),
        })
        .collect::<Vec<_>>();
    let omitted_line_count = sections.iter().map(|s| s.omitted_occurrences).sum();
    Ok(EvidenceLogDigest {
        format: "perseus_vault_evidence_log_digest",
        source_sha256: sha256.to_string(),
        config_version: DIGEST_VERSION,
        input_line_count: total,
        omitted_line_count,
        protected_line_count: protected_lines.len(),
        sections,
        protected_lines,
        retrieval:
            "Use perseus_vault_artifact_excerpt with returned anchors for exact original bytes.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn anchor(s: usize, e: usize) -> ArtifactAnchor {
        ArtifactAnchor {
            sha256: "a".repeat(64),
            byte_start: s as i64,
            byte_end: e as i64,
            line_start: Some(1),
            line_end: Some(1),
        }
    }
    #[test]
    fn deterministic_counts_and_protected_lines_are_preserved() {
        let input = b"INFO job=100 complete\nINFO job=101 complete\nERROR connection timeout id=22\nINFO job=102 complete\n";
        let a = digest(&"a".repeat(64), input, anchor).unwrap();
        let b = digest(&"a".repeat(64), input, anchor).unwrap();
        assert_eq!(
            serde_json::to_vec(&a).unwrap(),
            serde_json::to_vec(&b).unwrap()
        );
        assert_eq!(a.protected_line_count, 1);
        assert_eq!(a.protected_lines[0].0, "ERROR connection timeout id=22");
        assert_eq!(a.sections[0].count, 3);
        assert_eq!(a.sections[0].omitted_occurrences, 1);
    }

    #[test]
    fn ci_deploy_service_and_repeat_fixtures_keep_every_protected_line() {
        let cases = [
            b"CI step=1 passed\nCI step=2 passed\nWARN cache miss\n".as_slice(),
            b"deploy id=1 complete\ndeploy id=2 complete\nFATAL rollout refused\n".as_slice(),
            b"service worker=1 healthy\nservice worker=2 healthy\npanic: unexpected state\nTraceback (most recent call last):\n".as_slice(),
            b"request id=1 ok\nrequest id=2 ok\nDENIED token invalid\nASSERTION failed\nEXCEPTION failed\n".as_slice(),
        ];
        for input in cases {
            let digest = digest(&"b".repeat(64), input, anchor).unwrap();
            let source = std::str::from_utf8(input).unwrap();
            let protected_source = source.lines().filter(|line| protected(line)).count();
            assert_eq!(digest.protected_line_count, protected_source);
            assert_eq!(digest.protected_lines.len(), protected_source);
            for (line, _) in &digest.protected_lines {
                assert!(
                    source.contains(line),
                    "protected line must be verbatim: {line}"
                );
            }
            let collapsed_count: usize = digest.sections.iter().map(|s| s.count).sum();
            assert_eq!(
                collapsed_count + digest.protected_line_count,
                source.lines().count()
            );
        }
    }
}
