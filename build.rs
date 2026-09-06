use std::path::Path;

fn main() {
    // #713: embed the git commit hash for build identity reporting.
    embed_git_hash();

    #[cfg(feature = "grpc")]
    {
        tonic_build::configure()
            .build_server(true)
            .build_client(false)
            .compile_protos(&["proto/perseus_vault/v1/perseus_vault.proto"], &["proto"])
            .expect("failed to compile perseus-vault proto");
    }

    // #237: when bundled-embeddings is active, fetch the quantized
    // all-MiniLM-L6-v2 model + tokenizer once into OUT_DIR so embedding.rs can
    // `include_bytes!` them into the binary. The result is a single self-contained
    // binary that does dense/hybrid search with zero network at runtime. Cached in
    // OUT_DIR across incremental builds; a clean build downloads ~23MB once.
    if std::env::var("CARGO_FEATURE_BUNDLED_EMBEDDINGS").is_ok() {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        fetch_model_assets(&out_dir);
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=GIT_HASH");
    println!("cargo:rerun-if-changed=.git/HEAD");
    // Also track the current ref so detached-HEAD builds still get the hash.
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        if let Some(ref_path) = head.strip_prefix("ref: ") {
            let ref_path = ref_path.trim();
            println!("cargo:rerun-if-changed=.git/{}", ref_path);
        }
    }
}

/// Embed the git commit hash as GIT_HASH at compile time. Falls back to
/// "unknown" when .git is absent (e.g. source tarball builds), so the binary
/// always reports a build identity.
fn embed_git_hash() {
    let hash = git_describe_hash().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GIT_HASH={}", hash);
}

fn git_describe_hash() -> Option<String> {
    // Prefer an explicit env override (docker builds, CI where .git may be shallow).
    if let Ok(h) = std::env::var("GIT_HASH") {
        if !h.is_empty() {
            return Some(h);
        }
    }
    // Try `git describe --always --dirty` for a human-readable identity.
    let output = std::process::Command::new("git")
        .args(["describe", "--always", "--dirty", "--long"])
        .output()
        .ok()?;
    if output.status.success() {
        let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    // Fallback: `git rev-parse HEAD` for just the hash.
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if output.status.success() {
        return String::from_utf8_lossy(&output.stdout)
            .trim()
            .to_string()
            .into();
    }
    None
}

#[allow(dead_code)]
fn fetch_model_assets(out_dir: &str) {
    // The int8 dynamic-quantized ONNX export (~23MB vs ~90MB fp32; 384-dim, recall
    // within noise) plus its tokenizer, from the SAME sentence-transformers repo as
    // the fp32 model — so it keeps the (input_ids, attention_mask) -> last_hidden_state
    // signature the inference code already handles. The qint8 ops run on any CPU via
    // ONNX Runtime's CPU EP (the arch suffix is just the export's calibration preset).
    //
    // Supply-chain pinning: the URL is anchored to an IMMUTABLE commit revision
    // (not the mutable `main` ref), and every asset is SHA-256 verified before it
    // is baked into the binary via include_bytes!. A compromised or merely updated
    // upstream repo therefore cannot silently change the embedded model — a
    // mismatch fails the build loudly. The hashes are the HuggingFace LFS oids for
    // this revision (reproducible with `sha256sum` on the downloaded files).
    const REV: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
    let model_url = format!(
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{REV}/onnx/model_qint8_avx512_vnni.onnx"
    );
    let tokenizer_url = format!(
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{REV}/tokenizer.json"
    );
    const MODEL_SHA256: &str = "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474";
    const TOKENIZER_SHA256: &str =
        "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037";

    let model_dest = format!("{out_dir}/model_quantized.onnx");
    let tokenizer_dest = format!("{out_dir}/tokenizer.json");

    // Allow an operator/CI to pre-place or override the model dir (offline builds,
    // air-gapped CI) instead of downloading. Overridden files are still checksum-
    // verified below by ensure_asset, so an offline build can't be tricked into
    // embedding a mismatched model either.
    if let Ok(dir) = std::env::var("PERSEUS_VAULT_BUNDLED_MODEL_DIR") {
        copy_if_present(&dir, "model_quantized.onnx", out_dir);
        copy_if_present(&dir, "tokenizer.json", out_dir);
    }
    ensure_asset(&model_url, &model_dest, MODEL_SHA256);
    ensure_asset(&tokenizer_url, &tokenizer_dest, TOKENIZER_SHA256);
}

#[allow(dead_code)]
fn copy_if_present(src_dir: &str, name: &str, out_dir: &str) {
    let src = Path::new(src_dir).join(name);
    if src.exists() {
        let _ = std::fs::copy(&src, Path::new(out_dir).join(name));
    }
}

/// Ensure `dest` exists and matches `expected_sha256`, downloading from `url` if
/// needed. A cached (or operator-supplied) file whose hash matches is reused; any
/// other case re-downloads and verifies. The SHA-256 check subsumes the old
/// min-bytes truncation heuristic (a truncated download simply won't match).
#[allow(dead_code)]
fn ensure_asset(url: &str, dest: &str, expected_sha256: &str) {
    if let Ok(bytes) = std::fs::read(dest) {
        if sha256_hex(&bytes) == expected_sha256 {
            return;
        }
        // Stale cache or mismatched override — fall through and re-fetch.
    }
    let resp = ureq::get(url)
        .timeout(std::time::Duration::from_secs(600))
        .call()
        .unwrap_or_else(|e| panic!("build.rs: failed to download {url}: {e}\nFor an offline build, set PERSEUS_VAULT_BUNDLED_MODEL_DIR to a dir containing the model + tokenizer (still checksum-verified), or build with --no-default-features."));
    let mut reader = resp.into_reader();
    let mut buf = Vec::new();
    std::io::copy(&mut reader, &mut buf)
        .unwrap_or_else(|e| panic!("build.rs: download read failed for {url}: {e}"));
    let got = sha256_hex(&buf);
    assert_eq!(
        got, expected_sha256,
        "build.rs: SHA-256 mismatch for {url}\n  expected {expected_sha256}\n  got      {got}\nRefusing to embed an unverified asset."
    );
    let tmp = format!("{dest}.tmp");
    std::fs::write(&tmp, &buf).unwrap_or_else(|e| panic!("build.rs: cannot write {tmp}: {e}"));
    std::fs::rename(&tmp, dest).unwrap_or_else(|e| panic!("build.rs: cannot finalize {dest}: {e}"));
}

#[allow(dead_code)]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}
