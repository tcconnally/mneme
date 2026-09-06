//! Local multimodal document text extraction (#236).
//!
//! Turns a document file into plain text for storage in memory, entirely
//! locally — no cloud parsing API, no network — preserving Perseus Vault's air-gapped
//! ethos. Plaintext / markdown / structured-text formats work in any build;
//! **DOCX and PDF extraction is behind the optional `multimodal` feature** so the
//! lean default binary stays dependency-free. Without the feature, requesting a
//! docx/pdf returns a clear "rebuild with --features multimodal" error rather
//! than failing opaquely.
//!
//! Two size bounds guard against denial-of-service: `PERSEUS_VAULT_MAX_INGEST_BYTES`
//! caps the on-disk file, and `PERSEUS_VAULT_MAX_DECOMPRESSED_BYTES` caps the bytes read
//! out of a DOCX (a DEFLATE zip whose entries can decompress far past the on-disk
//! cap — a zip bomb). See `extract_docx_limited`.

use std::path::Path;

/// Extensions read directly as UTF-8 text — no feature, no dependency.
const PLAINTEXT_EXTS: &[&str] = &[
    "txt", "md", "markdown", "rst", "csv", "tsv", "json", "jsonl", "yaml", "yml", "toml", "log",
    "xml", "htm", "html", "tex", "org",
];

/// Default cap on the on-disk size of a file ingested via `perseus_vault_ingest_file`,
/// overridable with `PERSEUS_VAULT_MAX_INGEST_BYTES` (bytes; 0/invalid → default). The
/// extracted text is materialized in memory and then copied into a
/// JSON body and the FTS index, so an unbounded read is an OOM / denial-of-service
/// vector. Enforced before any file read (and before zip/pdf parsing, which can
/// amplify memory further).
const DEFAULT_MAX_INGEST_BYTES: u64 = 50 * 1024 * 1024; // 50 MiB

fn max_ingest_bytes() -> u64 {
    std::env::var("PERSEUS_VAULT_MAX_INGEST_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_INGEST_BYTES)
}

/// Default cap on the DECOMPRESSED size of a container entry (currently a DOCX's
/// `word/document.xml`), overridable with `PERSEUS_VAULT_MAX_DECOMPRESSED_BYTES`. This is
/// a SEPARATE limit from `max_ingest_bytes`: a DOCX is a DEFLATE zip, so a small
/// on-disk file (within the ingest cap) can decompress to many GB — a classic zip
/// bomb. The on-disk cap cannot bound that; this one does, by capping the bytes
/// read out of the archive. Generous headroom over any realistic document, a hard
/// ceiling against a bomb.
#[cfg(feature = "multimodal")]
const DEFAULT_MAX_DECOMPRESSED_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB

#[cfg(feature = "multimodal")]
fn max_decompressed_bytes() -> u64 {
    std::env::var("PERSEUS_VAULT_MAX_DECOMPRESSED_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_DECOMPRESSED_BYTES)
}

/// Reject a file whose on-disk size exceeds `max_bytes`, before it is read.
fn enforce_ingest_size(path: &Path, max_bytes: u64) -> Result<(), String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {}", path.display(), e))?
        .len();
    if len > max_bytes {
        return Err(format!(
            "{}: file is {} bytes, exceeding the {}-byte ingest limit \
             (raise PERSEUS_VAULT_MAX_INGEST_BYTES to override)",
            path.display(),
            len,
            max_bytes
        ));
    }
    Ok(())
}

/// Extract plain text from a document, routing by file extension.
pub fn extract_text(path: &Path) -> Result<String, String> {
    extract_text_limited(path, max_ingest_bytes())
}

/// Inner extractor with an explicit size cap (for deterministic testing without
/// touching the process environment).
fn extract_text_limited(path: &Path, max_bytes: u64) -> Result<String, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    if PLAINTEXT_EXTS.contains(&ext.as_str()) {
        enforce_ingest_size(path, max_bytes)?;
        return std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {}", path.display(), e));
    }

    match ext.as_str() {
        "docx" => extract_docx(path, max_bytes),
        "pdf" => extract_pdf(path, max_bytes),
        "" => Err(format!(
            "{}: no file extension; cannot determine document format",
            path.display()
        )),
        other => Err(format!(
            "unsupported document type '.{other}'. Supported: plaintext ({}); \
             docx/pdf require building with --features multimodal",
            PLAINTEXT_EXTS.join(", ")
        )),
    }
}

#[cfg(feature = "multimodal")]
fn extract_docx(path: &Path, max_bytes: u64) -> Result<String, String> {
    extract_docx_limited(path, max_bytes, max_decompressed_bytes())
}

/// DOCX extraction with an explicit decompressed-size cap (test seam).
///
/// Zip-bomb defense: `word/document.xml` is DEFLATE-compressed, so a tiny on-disk
/// entry (well within `max_bytes`) can decompress to many GB. We read at most
/// `max_decompressed + 1` bytes and reject if the entry is larger, instead of
/// letting `read_to_string` materialize an unbounded `String` (OOM). The entry's
/// declared uncompressed `size()` is deliberately IGNORED — it is attacker-
/// controlled metadata and can lie; the bound is enforced on the actual read.
#[cfg(feature = "multimodal")]
fn extract_docx_limited(
    path: &Path,
    max_bytes: u64,
    max_decompressed: u64,
) -> Result<String, String> {
    use std::io::Read;
    enforce_ingest_size(path, max_bytes)?;
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| format!("{} is not a valid .docx (zip): {e}", path.display()))?;
    let mut xml = String::new();
    {
        let doc = zip
            .by_name("word/document.xml")
            .map_err(|_| format!("{}: .docx is missing word/document.xml", path.display()))?;
        // Cap the decompressed read at max_decompressed + 1: if we manage to read
        // that many bytes, the entry exceeds the limit and is rejected.
        let mut limited = doc.take(max_decompressed.saturating_add(1));
        limited
            .read_to_string(&mut xml)
            .map_err(|e| format!("read word/document.xml: {e}"))?;
        if xml.len() as u64 > max_decompressed {
            return Err(format!(
                "{}: word/document.xml decompresses beyond the {}-byte limit \
                 (possible zip bomb; raise PERSEUS_VAULT_MAX_DECOMPRESSED_BYTES to override)",
                path.display(),
                max_decompressed
            ));
        }
    }
    Ok(docx_xml_to_text(&xml))
}

#[cfg(not(feature = "multimodal"))]
fn extract_docx(_path: &Path, _max_bytes: u64) -> Result<String, String> {
    Err("DOCX extraction requires building with --features multimodal".to_string())
}

#[cfg(feature = "multimodal")]
fn extract_pdf(path: &Path, max_bytes: u64) -> Result<String, String> {
    // The on-disk cap is the only bound available here: `pdf_extract` owns the
    // parse + FlateDecode decompression internally and exposes no streaming/limit
    // API, so a decompression-amplifying PDF is bounded only by its compressed
    // input size (max_bytes), not its decompressed size. Lower PERSEUS_VAULT_MAX_INGEST_BYTES
    // for untrusted PDF sources; a hard decompressed bound would require sandboxing
    // the parse (out of scope). DOCX, whose single entry we read ourselves, IS
    // decompressed-bounded — see extract_docx_limited.
    enforce_ingest_size(path, max_bytes)?;
    pdf_extract::extract_text(path)
        .map_err(|e| format!("PDF text extraction failed for {}: {e}", path.display()))
}

#[cfg(not(feature = "multimodal"))]
fn extract_pdf(_path: &Path, _max_bytes: u64) -> Result<String, String> {
    Err("PDF extraction requires building with --features multimodal".to_string())
}

/// Strip WordprocessingML to readable text: keep `<w:t>` runs, turn paragraph
/// (`</w:p>`), break (`<w:br>`) and tab (`<w:tab>`) tags into whitespace, drop
/// every other tag. Dependency-free and always compiled (so it is unit-testable
/// without the `multimodal` feature) — good enough for memory ingestion, where we
/// want the text, not faithful layout.
#[allow(dead_code)] // live under the `multimodal` feature + always unit-tested
pub fn docx_xml_to_text(xml: &str) -> String {
    let mut out = String::new();
    let mut chars = xml.char_indices().peekable();
    let mut capture = false; // inside a <w:t> ... </w:t> run

    while let Some((_, c)) = chars.next() {
        if c == '<' {
            // Read the tag up to '>'.
            let mut tag = String::new();
            for (_, tc) in chars.by_ref() {
                if tc == '>' {
                    break;
                }
                tag.push(tc);
            }
            let name = tag.trim_start_matches('/').trim();
            // Match the element name irrespective of attributes. A bare
            // self-closing tag with no attributes (e.g. `<w:tab/>`, which is
            // exactly what real Word documents emit for tab stops and manual
            // line breaks) has no whitespace before the `/`, so split on
            // whitespace alone would yield "w:tab/" and never match "w:tab"
            // below — strip a trailing '/' too.
            let elem = name
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches('/');
            if elem == "w:t" {
                capture = !tag.starts_with('/');
            } else if tag.starts_with('/') && elem == "w:p" {
                out.push('\n');
            } else if elem == "w:br" {
                out.push('\n');
            } else if elem == "w:tab" {
                out.push('\t');
            }
        } else if capture {
            out.push(c);
        }
    }

    unescape_xml(out.trim()).to_string()
}

/// Minimal XML entity unescape for the five predefined entities.
#[allow(dead_code)] // called by docx_xml_to_text (multimodal feature + tests)
fn unescape_xml(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&") // last, so we don't double-decode
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn plaintext_is_read_directly() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("perseus_vault-mm-{}.md", uuid::Uuid::new_v4()));
        std::fs::File::create(&p)
            .unwrap()
            .write_all(b"# Title\n\nbody text")
            .unwrap();
        let text = extract_text(&p).unwrap();
        assert!(text.contains("body text"));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn oversized_file_is_rejected_before_read() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("perseus_vault-mm-{}.txt", uuid::Uuid::new_v4()));
        std::fs::File::create(&p)
            .unwrap()
            .write_all(b"hello world, this is more than ten bytes of text")
            .unwrap();

        // A cap smaller than the file is rejected with a clear, actionable error.
        let err = extract_text_limited(&p, 10).unwrap_err();
        assert!(err.contains("ingest limit"), "got: {err}");
        assert!(err.contains("PERSEUS_VAULT_MAX_INGEST_BYTES"), "got: {err}");

        // A generous cap reads normally.
        let ok = extract_text_limited(&p, 10_000).unwrap();
        assert!(ok.contains("hello world"));

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn unsupported_extension_errors_clearly() {
        let err = extract_text(Path::new("/x/foo.xyz")).unwrap_err();
        assert!(err.contains("unsupported document type"));
    }

    #[test]
    fn no_extension_errors() {
        assert!(extract_text(Path::new("/x/README"))
            .unwrap_err()
            .contains("no file extension"));
    }

    #[test]
    fn docx_xml_to_text_extracts_runs_and_paragraphs() {
        let xml = r#"<w:document><w:body>
            <w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t xml:space="preserve"> world</w:t></w:r></w:p>
            <w:p><w:r><w:t>Line &amp; two</w:t></w:r></w:p>
            </w:body></w:document>"#;
        let text = docx_xml_to_text(xml);
        assert!(text.contains("Hello world"), "got: {text:?}");
        assert!(text.contains("Line & two"), "got: {text:?}");
        // Paragraph boundary becomes a newline.
        assert!(text.contains('\n'));
    }

    #[test]
    fn docx_ignores_non_text_tags() {
        let xml = "<w:p><w:pPr><w:spacing/></w:pPr><w:r><w:t>kept</w:t></w:r></w:p>";
        assert_eq!(docx_xml_to_text(xml), "kept");
    }

    #[test]
    fn docx_bare_self_closing_tab_and_br_are_not_dropped() {
        // Bare (no-attribute) self-closing <w:tab/> and <w:br/> are exactly what
        // real Word documents emit for tab stops and manual line breaks — the
        // tag name previously retained the trailing '/' and never matched
        // "w:tab"/"w:br", silently concatenating the neighboring runs together.
        let xml = "<w:p><w:r><w:t>Col1</w:t><w:tab/><w:t>Col2</w:t></w:r></w:p>";
        assert_eq!(docx_xml_to_text(xml), "Col1\tCol2");

        let xml = "<w:p><w:r><w:t>Line1</w:t><w:br/><w:t>Line2</w:t></w:r></w:p>";
        assert_eq!(docx_xml_to_text(xml), "Line1\nLine2");

        // Self-closing WITH attributes already worked (whitespace precedes the
        // '/'); keep it covered so a future change can't regress it silently.
        let xml =
            "<w:p><w:r><w:t>A</w:t><w:tab w:val=\"left\" w:pos=\"720\"/><w:t>B</w:t></w:r></w:p>";
        assert_eq!(docx_xml_to_text(xml), "A\tB");
    }

    #[cfg(not(feature = "multimodal"))]
    #[test]
    fn docx_pdf_error_without_feature() {
        assert!(extract_text(Path::new("/x/a.docx"))
            .unwrap_err()
            .contains("--features multimodal"));
        assert!(extract_text(Path::new("/x/a.pdf"))
            .unwrap_err()
            .contains("--features multimodal"));
    }

    #[cfg(feature = "multimodal")]
    #[test]
    fn docx_roundtrip_via_zip() {
        use zip::write::SimpleFileOptions;
        let dir = std::env::temp_dir();
        let p = dir.join(format!("perseus_vault-mm-{}.docx", uuid::Uuid::new_v4()));
        {
            let file = std::fs::File::create(&p).unwrap();
            let mut zw = zip::ZipWriter::new(file);
            zw.start_file("word/document.xml", SimpleFileOptions::default())
                .unwrap();
            zw.write_all(
                b"<w:document><w:body><w:p><w:r><w:t>Hello from docx</w:t></w:r></w:p></w:body></w:document>",
            )
            .unwrap();
            zw.finish().unwrap();
        }
        let text = extract_text(&p).unwrap();
        assert!(text.contains("Hello from docx"), "got: {text:?}");
        let _ = std::fs::remove_file(&p);
    }

    #[cfg(feature = "multimodal")]
    #[test]
    fn docx_decompression_bomb_is_rejected() {
        use zip::write::SimpleFileOptions;
        let dir = std::env::temp_dir();
        let p = dir.join(format!(
            "perseus_vault-mm-bomb-{}.docx",
            uuid::Uuid::new_v4()
        ));
        {
            let file = std::fs::File::create(&p).unwrap();
            let mut zw = zip::ZipWriter::new(file);
            // Highly repetitive, DEFLATE-compressed: ~5 MiB decompressed but only
            // a few KB on disk — passes any sane on-disk ingest cap while blowing
            // a small decompressed cap. This is the zip-bomb shape.
            zw.start_file(
                "word/document.xml",
                SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
            zw.write_all(&vec![b'A'; 5 * 1024 * 1024]).unwrap();
            zw.finish().unwrap();
        }

        // On-disk size is tiny; a generous ingest cap (50 MiB) passes, but the
        // entry decompresses past a 64 KiB decompressed cap -> rejected as a bomb.
        let err = extract_docx_limited(&p, 50 * 1024 * 1024, 64 * 1024).unwrap_err();
        assert!(err.contains("decompresses beyond"), "got: {err}");
        assert!(
            err.contains("PERSEUS_VAULT_MAX_DECOMPRESSED_BYTES"),
            "got: {err}"
        );

        // A decompressed cap above the entry size extracts without error.
        let ok = extract_docx_limited(&p, 50 * 1024 * 1024, 16 * 1024 * 1024);
        assert!(
            ok.is_ok(),
            "generous decompressed cap should succeed: {ok:?}"
        );

        let _ = std::fs::remove_file(&p);
    }
}
