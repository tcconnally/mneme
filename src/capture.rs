//! #520: opt-in in-session memory capture — distill a transcript / insight
//! payload into durable memory entities the moment a problem is solved,
//! instead of waiting for a scheduled harvest to notice.
//!
//! This module is the **distiller**: pure, deterministic text → candidate
//! notes, with zero network / zero LLM by default (the same local-first bar
//! as `extraction.rs`, which handles sentence-level fact/preference items;
//! this handles note-level session takeaways). An optional LLM path exists
//! at the tool layer (`tools::handle_capture` with `llm: true`) and falls
//! back here on any failure — the rule-based distiller is the floor, not a
//! degraded mode.
//!
//! Pipeline: [`split_candidates`] (headed sections / paragraphs / JSONL) →
//! [`classify`] (root-cause / pitfall / decision / pattern / takeaway via
//! cheap keyword markers, failure markers aligned with the #521 deja-vu
//! guard) → [`summary_line`] + [`key_for`] (stable slug key) → capped
//! [`DistillReport`]. Writing the notes (with trigram near-dup merging ON —
//! that is the anti-flood control) happens in `tools::handle_capture`.

use serde::Serialize;

// #888: span-of-origin hashing for source-chunk expansion.
use sha2::{Digest, Sha256};

/// Hard cap on entities written per capture invocation (anti-flood, #520).
/// Callers can lower it per call; they cannot raise it.
pub const MAX_CAPTURE_NOTES: usize = 20;

/// Candidates shorter than this (in chars, after trimming) are discarded as
/// non-durable chatter ("ok", "done", "thanks"). Precision over recall.
const MIN_CANDIDATE_CHARS: usize = 16;

/// Max length of the extracted summary line (chars).
const MAX_SUMMARY_CHARS: usize = 160;

/// Max length of the slugified key (chars).
const MAX_KEY_CHARS: usize = 64;

/// A character-offset span into the ORIGINAL capture payload (char offsets,
/// not bytes — safe on any UTF-8). `end_char` is exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CharSpan {
    pub start_char: usize,
    pub end_char: usize,
}

/// A single distilled, durable note ready to be remembered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CaptureNote {
    /// One of [`CAPTURE_ENTITY_TYPES`].
    pub entity_type: String,
    /// Stable slug key derived from the summary — the same solved problem
    /// re-captured with the same headline updates in place instead of
    /// creating a sibling row.
    pub key: String,
    /// One-line summary (the key basis and the recall headline).
    pub summary: String,
    /// The full candidate note text.
    pub content: String,
    /// #888: the char span of this note inside the payload it was distilled
    /// from. `Some` only on the rule-based path (LLM output has no reliable
    /// span); the writer stamps `source_chunk` refs only for spanned notes.
    pub span: Option<CharSpan>,
}

/// The distiller's output: capped notes plus accounting for what was dropped.
#[derive(Debug, Clone, Serialize)]
pub struct DistillReport {
    pub notes: Vec<CaptureNote>,
    /// Candidates that survived the minimum-length filter, before capping.
    pub candidates: usize,
    /// Candidates dropped by the per-invocation cap (logged, not silent).
    pub dropped: usize,
}

/// The closed set of entity types a capture may write. LLM output is
/// validated against this list (anything else degrades to "takeaway") —
/// model output is untrusted data, same rule as `dream`'s insight types.
pub const CAPTURE_ENTITY_TYPES: [&str; 5] =
    ["root-cause", "pitfall", "decision", "pattern", "takeaway"];

// ─── Classification markers ─────────────────────────────────────
//
// All lowercase substring markers, matched against the lowercased note.
// Priority: root-cause > pitfall > decision > pattern > takeaway — a note
// that names a failure AND its cause is a root-cause; a failure without a
// cause is a pitfall.

/// Markers that a note explains WHY something failed (diagnosis, not just
/// symptom). Checked before the failure markers.
const ROOT_CAUSE_MARKERS: &[&str] = &[
    "root cause",
    "root-cause",
    "caused by",
    " because ",
    "turned out",
    "the fix was",
    "the culprit",
    "traced to",
    "due to",
];

/// Markers that a note DESCRIBES A FAILURE. Kept aligned with the #521
/// deja-vu guard's `db::FAILURE_MARKERS` (same substring semantics: "fail"
/// covers failed/failure/failing; "bug" deliberately excluded because
/// "debug" false-positives on routine payloads) so a captured pitfall is
/// findable by `perseus_vault_check_failure_pattern`. The root-cause-only markers
/// live in [`ROOT_CAUSE_MARKERS`] and win first.
const FAILURE_MARKERS: &[&str] = &[
    "fail",
    "error",
    "pitfall",
    "broke",
    "mistake",
    "wrong",
    "regression",
    "doesn't work",
    "does not work",
    "didn't work",
    "did not work",
    "incident",
    "postmortem",
];

/// Markers of a committed choice between alternatives.
const DECISION_MARKERS: &[&str] = &[
    "decided",
    "decision",
    "chose",
    "we will",
    "going with",
    "opted",
    "instead of",
    "standing rule",
    "agreed to",
];

/// Markers of a reusable recipe / convention.
const PATTERN_MARKERS: &[&str] = &[
    "pattern",
    "whenever",
    "recipe",
    "workflow",
    "convention",
    "rule of thumb",
    "lesson",
    "always ",
    "works: ",
];

/// Classify a candidate note into one of [`CAPTURE_ENTITY_TYPES`].
pub fn classify(text: &str) -> &'static str {
    let lower = text.to_lowercase();
    if ROOT_CAUSE_MARKERS.iter().any(|m| lower.contains(m)) {
        return "root-cause";
    }
    if FAILURE_MARKERS.iter().any(|m| lower.contains(m)) {
        return "pitfall";
    }
    if DECISION_MARKERS.iter().any(|m| lower.contains(m)) {
        return "decision";
    }
    if PATTERN_MARKERS.iter().any(|m| lower.contains(m)) {
        return "pattern";
    }
    "takeaway"
}

/// Truncate to at most `max` chars (not bytes — safe on any UTF-8).
pub fn clip_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

/// First non-empty line of the note, stripped of markdown lead-in
/// (`#`/`-`/`*`/`>` and whitespace), clipped to [`MAX_SUMMARY_CHARS`].
pub fn summary_line(text: &str) -> String {
    for line in text.lines() {
        let stripped = line
            .trim_start_matches(|c: char| {
                c == '#' || c == '-' || c == '*' || c == '>' || c.is_whitespace()
            })
            .trim();
        if !stripped.is_empty() {
            return clip_chars(stripped, MAX_SUMMARY_CHARS);
        }
    }
    String::new()
}

/// FNV-1a over the input — a tiny stable hash for fallback keys. No crypto
/// claim; only used to make a non-sluggable summary produce a stable key.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Slugify a summary into a stable ASCII key: lowercase, alphanumeric runs
/// joined by single `-`, clipped to [`MAX_KEY_CHARS`]. A summary with no
/// ASCII-alphanumeric content (emoji-only, CJK, …) falls back to a stable
/// `note-<hash>` key so it still round-trips deterministically.
pub fn key_for(summary: &str) -> String {
    let mut slug = String::with_capacity(summary.len());
    let mut last_dash = true; // suppress a leading dash
    for c in summary.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
        if slug.len() >= MAX_KEY_CHARS {
            break;
        }
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        format!("note-{:016x}", fnv1a(summary))
    } else {
        slug
    }
}

/// True when every non-empty line parses as a JSON object — the JSONL shape
/// hook payloads and transcript exports commonly use.
fn looks_like_jsonl(payload: &str) -> bool {
    let mut saw_any = false;
    for line in payload.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        saw_any = true;
        if !t.starts_with('{')
            || serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(t).is_err()
        {
            return false;
        }
    }
    saw_any
}

/// Pull the note text out of one JSONL record: the first present non-empty
/// string among the conventional content fields, else the compact record
/// itself (still classifiable — markers survive JSON encoding).
fn jsonl_note_text(record: &serde_json::Map<String, serde_json::Value>) -> String {
    const CONTENT_FIELDS: &[&str] = &["content", "text", "insight", "lesson", "summary", "message"];
    for field in CONTENT_FIELDS {
        if let Some(serde_json::Value::String(s)) = record.get(*field) {
            let t = s.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    serde_json::Value::Object(record.clone()).to_string()
}

/// Trim a raw char range of `payload` to its non-whitespace core, returning
/// the trimmed text and the correspondingly adjusted span. `None` when the
/// range is out of bounds or empty after trimming.
fn trimmed_span(payload: &str, start_char: usize, end_char: usize) -> Option<(String, CharSpan)> {
    let total = payload.chars().count();
    if start_char > end_char || end_char > total {
        return None;
    }
    let mut it = payload.char_indices();
    let start_byte = it.nth(start_char).map(|(b, _)| b)?;
    let end_byte = if end_char == start_char {
        start_byte
    } else {
        match it.nth(end_char - start_char - 1) {
            Some((b, _)) => b,
            None => payload.len(),
        }
    };
    let raw = &payload[start_byte..end_byte];
    let raw_chars = raw.chars().count();
    let lead = raw.chars().take_while(|c| c.is_whitespace()).count();
    let trail = raw.chars().rev().take_while(|c| c.is_whitespace()).count();
    if lead + trail >= raw_chars {
        return None;
    }
    let content: String = raw
        .chars()
        .skip(lead)
        .take(raw_chars - lead - trail)
        .collect();
    Some((
        content,
        CharSpan {
            start_char: start_char + lead,
            end_char: end_char - trail,
        },
    ))
}

/// Extract the verbatim text of `span` from `payload` (char offsets).
/// `None` when the span is out of bounds.
pub fn span_text<'a>(payload: &'a str, span: CharSpan) -> Option<&'a str> {
    let total = payload.chars().count();
    if span.start_char > span.end_char || span.end_char > total {
        return None;
    }
    let mut it = payload.char_indices();
    let start_byte = it.nth(span.start_char).map(|(b, _)| b)?;
    let end_byte = if span.end_char == span.start_char {
        start_byte
    } else {
        match it.nth(span.end_char - span.start_char - 1) {
            Some((b, _)) => b,
            None => payload.len(),
        }
    };
    Some(&payload[start_byte..end_byte])
}

/// SHA-256 hex digest of the verbatim span text — the integrity anchor the
/// expand surface re-verifies against the retained transcript store.
pub fn span_sha256(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    format!("{:x}", h.finalize())
}

/// Stable key for the retained transcript entity: identical payloads
/// re-captured update the SAME transcript row (anti-flood, like notes).
pub fn transcript_key(payload: &str) -> String {
    format!("transcript-{:016x}", fnv1a(payload))
}

/// Split a payload into candidate note texts WITH their char spans.
///
/// Same three shapes as [`split_candidates`] (JSONL / headed markdown /
/// plain paragraphs). Each returned span points at the trimmed candidate
/// inside the ORIGINAL payload, so a writer can later re-extract the
/// verbatim text ([`span_text`]) and hash it for integrity verification.
///
/// Candidates shorter than [`MIN_CANDIDATE_CHARS`] are discarded.
pub fn split_candidates_spanned(payload: &str) -> Vec<(String, CharSpan)> {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // Walk the ORIGINAL payload line by line, tracking char offsets, so
    // spans survive any trimming. `pos` = char offset of the current line's
    // first char; a line's raw char range is [pos, pos + line_chars).
    let mut ranges: Vec<(usize, usize)> = Vec::new(); // (start_char, end_char)
    let mut pos: usize = 0;
    let mut current_start: Option<usize> = None;
    let mut current_end = 0usize;
    let mut headed = false;
    let jsonl = looks_like_jsonl(trimmed);

    for line in payload.split('\n') {
        let line_chars = line.chars().count();
        let line_start = pos;
        let line_end = pos + line_chars;
        pos += line_chars + 1; // +1 for the '\n'

        if jsonl {
            // One candidate per parseable JSON-object line.
            let t = line.trim();
            if !t.is_empty() {
                if let Ok(rec) =
                    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(t)
                {
                    if !jsonl_note_text(&rec).is_empty() {
                        ranges.push((line_start, line_end));
                    }
                }
            }
            continue;
        }
        if line.trim_start().starts_with('#') {
            headed = true;
            if let Some(s) = current_start.take() {
                ranges.push((s, current_end));
            }
            current_start = Some(line_start);
        } else if current_start.is_none() {
            current_start = Some(line_start);
        }
        current_end = line_end;
    }
    if let Some(s) = current_start.take() {
        ranges.push((s, current_end));
    }

    if !jsonl && !headed {
        // Plain text: blank-line-separated paragraphs.
        ranges.clear();
        let mut para_start: Option<usize> = None;
        let mut para_end = 0usize;
        let mut ppos = 0usize;
        for line in payload.split('\n') {
            let line_chars = line.chars().count();
            if line.trim().is_empty() {
                if let Some(s) = para_start.take() {
                    ranges.push((s, para_end));
                }
            } else if para_start.is_none() {
                para_start = Some(ppos);
                para_end = ppos + line_chars;
            } else {
                para_end = ppos + line_chars;
            }
            ppos += line_chars + 1;
        }
        if let Some(s) = para_start.take() {
            ranges.push((s, para_end));
        }
    }

    let mut out = Vec::new();
    for (s, e) in ranges {
        if let Some((content, span)) = trimmed_span(payload, s, e) {
            if content.chars().count() >= MIN_CANDIDATE_CHARS {
                out.push((content, span));
            }
        }
    }
    out
}

/// Split a payload into candidate note texts.
///
/// Three shapes, auto-detected:
/// 1. **JSONL** — every non-empty line is a JSON object → one candidate per
///    record (conventional content field, else the compact record).
/// 2. **Headed markdown** — any `#`-heading lines present → one candidate
///    per headed section (heading + body until the next heading); a
///    non-empty preamble before the first heading is its own candidate.
/// 3. **Plain text** — candidates are blank-line-separated paragraphs.
///
/// Candidates shorter than [`MIN_CANDIDATE_CHARS`] are discarded.
pub fn split_candidates(payload: &str) -> Vec<String> {
    split_candidates_spanned(payload)
        .into_iter()
        .map(|(text, _)| text)
        .collect()
}

/// The rule-based distiller: payload → classified, keyed, capped notes.
/// Deterministic and fully local (no LLM, no network, no DB access).
/// In-batch repeats of the same key keep the first occurrence (the DB-side
/// trigram dedup handles near-duplicates across invocations).
pub fn distill(payload: &str, max_notes: usize) -> DistillReport {
    let cap = max_notes.clamp(1, MAX_CAPTURE_NOTES);
    let candidates = split_candidates_spanned(payload);
    let total = candidates.len();

    let mut notes: Vec<CaptureNote> = Vec::new();
    for (candidate, span) in candidates {
        let summary = summary_line(&candidate);
        if summary.is_empty() {
            continue;
        }
        let key = key_for(&summary);
        if notes.iter().any(|n| n.key == key) {
            continue; // in-batch duplicate headline: first wins
        }
        notes.push(CaptureNote {
            entity_type: classify(&candidate).to_string(),
            key,
            summary,
            content: candidate,
            span: Some(span),
        });
    }

    let kept = notes.len().min(cap);
    let dropped = notes.len() - kept;
    notes.truncate(kept);
    DistillReport {
        notes,
        candidates: total,
        dropped,
    }
}

// ─── Optional LLM distillation (#520 `--llm`) ────────────────────
//
// The prompt/parse pair for the opt-in LLM path. The transport call itself
// lives on `Database` (`llm_generate`, gated on `llm_config.enabled` with
// the #528 PERSEUS_VAULT_LLM_TIMEOUT_SECS timeout); `tools::handle_capture` wires
// prompt → call → parse and falls back to [`distill`] on ANY failure.

/// Build the distillation prompt. Strict-JSON contract, same style as
/// `synthesize`'s lesson extraction.
pub fn llm_prompt(payload: &str) -> String {
    format!(
        r#"You are a memory distillation system for an AI agent. Given a session transcript or insight payload, extract the few durable notes worth remembering across sessions.

CRITICAL INSTRUCTIONS:
- Extract at most {max} notes; fewer is better. Only include notes that will still matter in a future session.
- Each note's "type" MUST be one of: "root-cause" (why something failed), "pitfall" (a failure to avoid), "decision" (a committed choice), "pattern" (a reusable recipe/convention), "takeaway" (anything else durable).
- "summary" is one line (max 160 chars); "content" is the full self-contained note.
- Return ONLY valid JSON. No markdown, no commentary.

Payload:
{payload}

Return a JSON object: {{"notes": [{{"type": "...", "summary": "...", "content": "..."}}]}}
If nothing is worth remembering, return: {{"notes": []}}"#,
        max = MAX_CAPTURE_NOTES,
        payload = payload
    )
}

/// Parse the LLM's distillation output. Tolerates a ```json fence; anything
/// else non-conforming returns `None` so the caller falls back to the
/// rule-based path. Unknown types degrade to "takeaway" (LLM output is
/// untrusted); empty content falls back to the summary.
pub fn parse_llm_notes(raw: &str, max_notes: usize) -> Option<DistillReport> {
    let cap = max_notes.clamp(1, MAX_CAPTURE_NOTES);
    let mut text = raw.trim();
    if let Some(stripped) = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
    {
        text = stripped.trim_start();
    }
    if let Some(stripped) = text.strip_suffix("```") {
        text = stripped.trim_end();
    }
    let parsed: serde_json::Value = serde_json::from_str(text).ok()?;
    let arr = parsed.get("notes")?.as_array()?;

    let mut notes: Vec<CaptureNote> = Vec::new();
    for item in arr {
        let summary = clip_chars(
            item.get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim(),
            MAX_SUMMARY_CHARS,
        );
        if summary.is_empty() {
            continue;
        }
        let content = item
            .get("content")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .unwrap_or(&summary)
            .to_string();
        let raw_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let entity_type = if CAPTURE_ENTITY_TYPES.contains(&raw_type) {
            raw_type.to_string()
        } else {
            "takeaway".to_string()
        };
        let key = key_for(&summary);
        if notes.iter().any(|n| n.key == key) {
            continue;
        }
        notes.push(CaptureNote {
            entity_type,
            key,
            summary,
            content,
            // LLM output has no reliable span into the payload; the writer
            // stamps no source_chunk for such notes (#888 graceful path).
            span: None,
        });
    }

    let total = notes.len();
    let kept = total.min(cap);
    let dropped = total - kept;
    notes.truncate(kept);
    Some(DistillReport {
        notes,
        candidates: total,
        dropped,
    })
}

// ─── #563: consume / prune-source ────────────────────────────────
//
// After a successful non-dry-run capture, remove exactly the captured regions
// from the source file so a write-buffer (e.g. an AGENTS.local.md that a host
// agent inlines every turn) does not accumulate durably-stored blocks forever.
// The removal is scoped to the captured records only; surrounding non-captured
// content (headers, rules, pointers) is left untouched.

/// Remove each captured note's region from `source`, returning the rewritten
/// text and the number of regions actually removed.
///
/// Matching is line-based and whitespace-tolerant: a note's region is the
/// contiguous run of source lines whose trimmed text equals the note's content
/// lines (the distiller trims candidates, so byte-equality on the raw source
/// would miss). Only lines belonging to a captured note are dropped — headers,
/// rules, and pointers that were never captured survive. Blank lines left
/// behind by a removal are collapsed so the file does not grow a run of gaps.
pub fn prune_captured_regions(source: &str, notes: &[CaptureNote]) -> (String, usize) {
    let src_lines: Vec<&str> = source.lines().collect();
    let mut removed_flags = vec![false; src_lines.len()];
    let mut removed = 0usize;

    for note in notes {
        let note_lines: Vec<&str> = note.content.lines().collect();
        if note_lines.is_empty() {
            continue;
        }
        // Find the first not-yet-removed window matching the note's lines.
        let n = note_lines.len();
        let mut found = None;
        if n <= src_lines.len() {
            for i in 0..=(src_lines.len() - n) {
                if (i..i + n).any(|k| removed_flags[k]) {
                    continue;
                }
                if (0..n).all(|k| src_lines[i + k].trim() == note_lines[k].trim()) {
                    found = Some(i);
                    break;
                }
            }
        }
        if let Some(i) = found {
            for k in i..i + n {
                removed_flags[k] = true;
            }
            removed += 1;
        }
    }

    if removed == 0 {
        return (source.to_string(), 0);
    }

    // Rebuild from the surviving lines, collapsing 2+ consecutive blank lines
    // (which a removal can create) down to one.
    let mut out_lines: Vec<&str> = Vec::with_capacity(src_lines.len());
    let mut prev_blank = false;
    for (idx, line) in src_lines.iter().enumerate() {
        if removed_flags[idx] {
            continue;
        }
        let is_blank = line.trim().is_empty();
        if is_blank && prev_blank {
            continue;
        }
        out_lines.push(line);
        prev_blank = is_blank;
    }
    // Drop leading/trailing blank lines introduced by boundary removals.
    while out_lines.first().is_some_and(|l| l.trim().is_empty()) {
        out_lines.remove(0);
    }
    while out_lines.last().is_some_and(|l| l.trim().is_empty()) {
        out_lines.pop();
    }

    let mut rebuilt = out_lines.join("\n");
    // Preserve a single trailing newline if the original had one and there is
    // still content.
    if !rebuilt.is_empty() && source.ends_with('\n') {
        rebuilt.push('\n');
    }
    (rebuilt, removed)
}

/// Read `path`, prune the captured regions, and — if anything was removed —
/// rewrite it atomically (temp file in the same directory + rename), leaving a
/// timestamped-content `<path>.bak` of the original. Returns the number of
/// regions removed (0 leaves the file, and any `.bak`, untouched).
///
/// This is the file side of `capture --consume`; it never runs under
/// `--dry-run` and is only called when the capture actually stored something,
/// so it can never delete source content that was not durably persisted.
pub fn consume_source_file(
    path: &std::path::Path,
    notes: &[CaptureNote],
) -> std::io::Result<usize> {
    let original = std::fs::read_to_string(path)?;
    let (pruned, removed) = prune_captured_regions(&original, notes);
    if removed == 0 {
        return Ok(0);
    }

    // Back up the original first, then write the pruned content atomically.
    let bak = {
        let mut p = path.as_os_str().to_os_string();
        p.push(".bak");
        std::path::PathBuf::from(p)
    };
    std::fs::write(&bak, original.as_bytes())?;

    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let tmp = dir.join(format!(
        ".{}.capture-tmp",
        path.file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| "source".to_string())
    ));
    std::fs::write(&tmp, pruned.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_headed_markdown_sections_with_preamble() {
        let payload = "Session context: fixed the deploy pipeline today.\n\n\
                       # Root cause of the deploy failure\nThe deploy failed because the schema version was never bumped.\n\n\
                       # Decision on toolchain\nWe decided to standardize on the MSVC toolchain for Windows builds.";
        let sections = split_candidates(payload);
        assert_eq!(sections.len(), 3, "{sections:?}");
        assert!(sections[0].starts_with("Session context"));
        assert!(sections[1].starts_with("# Root cause"));
        assert!(sections[2].starts_with("# Decision"));
    }

    #[test]
    fn splits_plain_paragraphs_and_drops_chatter() {
        let payload = "ok\n\nThe cargo build only works with the MSVC toolchain on Windows.\n\n\
                       done\n\nAlways trim PATH before invoking vcvars to avoid the 8191-char overflow.";
        let sections = split_candidates(payload);
        // "ok" and "done" are below the minimum-length filter.
        assert_eq!(sections.len(), 2, "{sections:?}");
    }

    #[test]
    fn splits_jsonl_records_via_content_fields() {
        let payload = r#"{"content": "The retry loop failed because the token expired mid-flight."}
{"text": "Decided to cache the token with a 5-minute refresh margin."}
{"kind": "misc", "note_id": 7, "detail": "record with no conventional field but plenty of length"}"#;
        let sections = split_candidates(payload);
        assert_eq!(sections.len(), 3, "{sections:?}");
        assert!(sections[0].contains("token expired"));
        assert!(sections[1].contains("refresh margin"));
        // Fallback: the compact record itself.
        assert!(sections[2].starts_with('{') && sections[2].contains("note_id"));
    }

    #[test]
    fn classifies_all_five_types() {
        assert_eq!(
            classify("The deploy failed; root cause was the unbumped schema version."),
            "root-cause"
        );
        assert_eq!(
            classify("The migration failed on the FK constraint."),
            "pitfall"
        );
        assert_eq!(
            classify("We decided to ship the fallback path first."),
            "decision"
        );
        assert_eq!(
            classify("Rule of thumb: run the smoke suite before every release."),
            "pattern"
        );
        assert_eq!(
            classify("The vault holds about 1,300 entities now."),
            "takeaway"
        );
    }

    #[test]
    fn root_cause_wins_over_pitfall() {
        // A failure WITH a diagnosis is a root-cause, not a pitfall.
        let text = "The build broke; turned out the linker needed vcvars in PATH.";
        assert_eq!(classify(text), "root-cause");
    }

    #[test]
    fn failure_markers_align_with_the_deja_vu_guard() {
        // Alignment pin (#521): every capture failure marker must be one the
        // deja-vu guard (`db::FAILURE_MARKERS`) also recognizes, so a
        // captured pitfall is findable by perseus_vault_check_failure_pattern. The
        // guard's list may be a superset (e.g. "root cause"/"root-cause"
        // live in ROOT_CAUSE_MARKERS here, which classify() checks first).
        for m in FAILURE_MARKERS {
            assert!(
                crate::db::FAILURE_MARKERS.contains(m),
                "capture failure marker {m:?} missing from db::FAILURE_MARKERS — \
                 keep the two lists aligned so captured pitfalls stay findable \
                 by the deja-vu guard"
            );
        }
    }

    #[test]
    fn summary_and_key_are_stable_and_bounded() {
        let text = "## The Fix: bump SCHEMA_VERSION on every new ensure_column!\nDetails follow.";
        let summary = summary_line(text);
        assert_eq!(
            summary,
            "The Fix: bump SCHEMA_VERSION on every new ensure_column!"
        );
        let key = key_for(&summary);
        assert_eq!(
            key,
            "the-fix-bump-schema-version-on-every-new-ensure-column"
        );
        // Deterministic.
        assert_eq!(key, key_for(&summary_line(text)));
        // Bounded + ASCII even for long/unicode summaries.
        let long_key = key_for(&"é🎉 ".repeat(200));
        assert!(long_key.starts_with("note-"), "{long_key}");
        assert!(key_for(&"x".repeat(500)).len() <= MAX_KEY_CHARS);
    }

    #[test]
    fn distill_caps_notes_and_reports_dropped() {
        let payload = (0..30)
            .map(|i| format!("Durable takeaway number {i} about the capture system."))
            .collect::<Vec<_>>()
            .join("\n\n");
        let report = distill(&payload, 50); // asks above the hard cap
        assert_eq!(report.candidates, 30);
        assert_eq!(report.notes.len(), MAX_CAPTURE_NOTES);
        assert_eq!(report.dropped, 10);

        // A caller can lower the cap, never raise it.
        let report = distill(&payload, 5);
        assert_eq!(report.notes.len(), 5);
        assert_eq!(report.dropped, 25);
    }

    #[test]
    fn distill_skips_in_batch_duplicate_keys() {
        let payload = "The build failed on the FK constraint again today.\n\n\
                       The build failed on the FK constraint again today.";
        let report = distill(&payload, 20);
        assert_eq!(report.notes.len(), 1, "{report:?}");
    }

    #[test]
    fn distill_is_deterministic() {
        let payload = "# Root cause\nThe deploy failed because of the stale cache.\n\n\
                       # Next step\nAlways invalidate the cache before deploying.";
        let a = distill(payload, 20);
        let b = distill(payload, 20);
        assert_eq!(a.notes, b.notes);
    }

    #[test]
    fn parse_llm_notes_happy_path_fences_and_junk() {
        let raw = r#"```json
{"notes": [
  {"type": "root-cause", "summary": "Token expiry broke retries", "content": "The retry loop failed because the token expired mid-flight."},
  {"type": "made-up-type", "summary": "Something else durable", "content": "Body."},
  {"type": "decision", "summary": ""}
]}
```"#;
        let report = parse_llm_notes(raw, 20).expect("fenced JSON must parse");
        assert_eq!(report.notes.len(), 2, "{report:?}"); // empty summary skipped
        assert_eq!(report.notes[0].entity_type, "root-cause");
        // Unknown type degrades to takeaway (untrusted LLM output).
        assert_eq!(report.notes[1].entity_type, "takeaway");

        // Junk → None (caller falls back to the rule-based distiller).
        assert!(parse_llm_notes("I could not find any notes, sorry!", 20).is_none());
        assert!(parse_llm_notes("{\"lessons\": []}", 20).is_none());
        // Missing content falls back to the summary.
        let report = parse_llm_notes(
            r#"{"notes": [{"type": "takeaway", "summary": "Just this"}]}"#,
            20,
        )
        .unwrap();
        assert_eq!(report.notes[0].content, "Just this");
    }

    // ─── #563 consume / prune-source ─────────────────────────────

    fn note(content: &str) -> CaptureNote {
        let summary = summary_line(content);
        CaptureNote {
            entity_type: classify(content).to_string(),
            key: key_for(&summary),
            summary,
            content: content.to_string(),
            span: None,
        }
    }

    #[test]
    fn prune_removes_only_captured_blocks_and_keeps_headers() {
        let source = "# AGENTS.local.md\n\
                      Standing rule: always run the smoke suite.\n\n\
                      ### 2026-07-10\n\
                      The deploy failed because the schema version was never bumped.\n\n\
                      ### 2026-07-11\n\
                      Decided to standardize on the MSVC toolchain for Windows builds.\n";
        // Capture the two dated blocks (as the distiller would).
        let notes = vec![
            note("### 2026-07-10\nThe deploy failed because the schema version was never bumped."),
            note(
                "### 2026-07-11\nDecided to standardize on the MSVC toolchain for Windows builds.",
            ),
        ];
        let (pruned, removed) = prune_captured_regions(source, &notes);
        assert_eq!(removed, 2, "both captured blocks must be removed");
        // Header + standing rule survive; captured bodies are gone.
        assert!(pruned.contains("# AGENTS.local.md"));
        assert!(pruned.contains("Standing rule"));
        assert!(!pruned.contains("schema version was never bumped"));
        assert!(!pruned.contains("MSVC toolchain"));
    }

    #[test]
    fn prune_is_a_noop_when_nothing_matches() {
        let source = "# Header\nUncaptured content only.\n";
        let notes = vec![note(
            "Something entirely different that is not in the file at all.",
        )];
        let (pruned, removed) = prune_captured_regions(source, &notes);
        assert_eq!(removed, 0);
        assert_eq!(pruned, source, "no match => source is returned verbatim");
    }

    #[test]
    fn consume_source_file_rewrites_atomically_and_backs_up() {
        let dir = std::env::temp_dir().join(format!("capture-consume-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("AGENTS.local.md");
        let source = "# Pointers\nSee the vault.\n\n\
                      ### 2026-07-10\n\
                      Root cause: the linker needed vcvars in PATH.\n";
        std::fs::write(&path, source).unwrap();

        let notes = vec![note(
            "### 2026-07-10\nRoot cause: the linker needed vcvars in PATH.",
        )];
        let removed = consume_source_file(&path, &notes).unwrap();
        assert_eq!(removed, 1);

        // Source no longer holds the captured block, but keeps the header.
        let after = std::fs::read_to_string(&path).unwrap();
        assert!(after.contains("# Pointers"));
        assert!(!after.contains("vcvars in PATH"));

        // A .bak with the ORIGINAL content exists.
        let bak = std::fs::read_to_string(dir.join("AGENTS.local.md.bak")).unwrap();
        assert_eq!(bak, source, "backup must hold the pre-prune original");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ─── #888: spanned splitting / span math ─────────────────────

    #[test]
    fn split_candidates_matches_the_spanned_wrapper() {
        let payload = "First paragraph about the failure and its root cause.\n\n\
                       Second paragraph about the decision taken.";
        let want = vec![
            "First paragraph about the failure and its root cause.".to_string(),
            "Second paragraph about the decision taken.".to_string(),
        ];
        assert_eq!(split_candidates(payload), want);
        let spanned = split_candidates_spanned(payload);
        assert_eq!(
            spanned.iter().map(|(t, _)| t.clone()).collect::<Vec<_>>(),
            want
        );
        // Every span re-extracts exactly its candidate (verbatim roundtrip).
        for (text, span) in &spanned {
            assert_eq!(span_text(payload, *span).unwrap(), text.as_str());
        }
    }

    #[test]
    fn spanned_char_offsets_are_utf8_safe() {
        // Multibyte content: offsets are CHARS, not bytes — a byte-counted
        // split would land mid-codepoint on the emoji character.
        let payload = "Key lesson: the deployment failed for lack of headroom, so the retry loop never recovered.\n\n\
                       🎉 WAL mode decision: we decided to use WAL for writes.\n\n\
                       Third paragraph about the cache invalidation decision.";
        let cands = split_candidates_spanned(payload);
        assert_eq!(cands.len(), 3, "{:?}", cands);
        for (text, span) in &cands {
            let verbatim = span_text(payload, *span).expect("span in bounds");
            assert_eq!(verbatim, text.as_str());
            assert!(verbatim.chars().count() >= MIN_CANDIDATE_CHARS);
        }
        // Paragraph spans are ordered and non-overlapping.
        let (_, s0) = cands[0];
        let (_, s1) = cands[1];
        let (_, s2) = cands[2];
        assert!(s0.start_char < s0.end_char);
        assert!(s1.start_char > s0.end_char, "{s0:?} vs {s1:?}");
        assert!(s2.start_char > s1.end_char, "{s1:?} vs {s2:?}");
        // The emoji paragraph begins at the emoji's CHAR offset — the byte
        // offset is larger because the emoji is 4 bytes. This pins the
        // char-vs-byte distinction that the splitter must preserve.
        let emoji_byte = payload.find("🎉").unwrap();
        assert_eq!(
            cands[1].1.start_char,
            payload[..emoji_byte].chars().count(),
            "start_char is a CHAR offset, not a byte offset"
        );
    }

    #[test]
    fn spanned_jsonl_keeps_per_line_ranges() {
        let payload = "{\"content\": \"first note about the token expiry\"}\n\
                       {\"content\": \"second note about the refresh margin\"}\n";
        let cands = split_candidates_spanned(payload);
        assert_eq!(cands.len(), 2);
        let (_, s0) = cands[0];
        let (_, s1) = cands[1];
        assert_eq!(s0.start_char, 0);
        assert!(s1.start_char > s0.end_char, "{s0:?} vs {s1:?}");
        assert_eq!(span_text(payload, s0).unwrap(), cands[0].0.as_str());
        assert_eq!(span_text(payload, s1).unwrap(), cands[1].0.as_str());
    }

    #[test]
    fn span_text_rejects_out_of_bounds_and_empty() {
        assert_eq!(
            span_text(
                "short",
                CharSpan {
                    start_char: 0,
                    end_char: 99
                }
            ),
            None
        );
        assert_eq!(
            span_text(
                "short",
                CharSpan {
                    start_char: 5,
                    end_char: 3
                }
            ),
            None
        );
        assert_eq!(
            span_text(
                "short",
                CharSpan {
                    start_char: 0,
                    end_char: 0
                }
            )
            .unwrap(),
            ""
        );
        assert_eq!(
            span_text(
                "short",
                CharSpan {
                    start_char: 2,
                    end_char: 5
                }
            )
            .unwrap(),
            "ort"
        );
    }

    #[test]
    fn distill_notes_carry_spans_only_on_the_rule_based_path() {
        let payload = "# Root cause\nThe deploy failed because of the stale cache.\n\n\
                       # Next step\nAlways invalidate the cache before deploying.";
        let report = distill(payload, 20);
        assert_eq!(report.notes.len(), 2);
        for note in &report.notes {
            let span = note.span.expect("rule-based notes must carry spans");
            assert_eq!(span_text(payload, span).unwrap(), note.content);
        }
        // LLM path: no spans (untrusted offsets).
        let llm = parse_llm_notes(
            r#"{"notes": [{"type": "takeaway", "summary": "Token expiry", "content": "Token expiry broke retries."}]}"#,
            20,
        )
        .unwrap();
        assert!(
            llm.notes[0].span.is_none(),
            "LLM notes must not fabricate spans"
        );
    }

    #[test]
    fn transcript_key_is_stable_and_span_hash_is_sha256() {
        let a = "same payload twice";
        let b = "same payload twice";
        let c = "different payload";
        assert_eq!(transcript_key(a), transcript_key(b));
        assert_ne!(transcript_key(a), transcript_key(c));
        let h = span_sha256("verbatim span text");
        assert_eq!(h.len(), 64);
        assert_eq!(h, span_sha256("verbatim span text"));
        assert_ne!(h, span_sha256("verbatim span text!"));
    }
}
