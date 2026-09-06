// Bundled embedding model — zero-dependency dense embeddings using ONNX Runtime.
//
// When --embedding-model is set, uses a local ONNX model for embeddings
// instead of requiring Ollama. The all-MiniLM-L6-v2 model (80MB, 384-dim) is
// downloaded on first use and cached at ~/.perseus-vault/models/.
//
// Inference backends:
//   - Native (feature = "bundled-embeddings"): ort + tokenizers crates
//   - Fallback: uses onnxruntime via Python subprocess (requires `pip install onnxruntime`)
//   - When neither is available, falls through to Ollama (if configured)

use std::path::PathBuf;

// #237: the quantized model + tokenizer are fetched by build.rs into OUT_DIR and
// compiled into the binary, so dense/hybrid search works fully offline with no
// first-run download. Only present when the (default-on) feature is enabled.
#[cfg(feature = "bundled-embeddings")]
static BUNDLED_MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model_quantized.onnx"));
#[cfg(feature = "bundled-embeddings")]
static BUNDLED_TOKENIZER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tokenizer.json"));

/// Configuration for the local embedding backend.
#[derive(Clone)]
pub struct EmbeddingConfig {
    /// Whether local embeddings are enabled.
    #[allow(dead_code)]
    pub enabled: bool,
    /// Path to the ONNX model file (used only when `bundled` is false).
    #[allow(dead_code)]
    pub model_path: PathBuf,
    /// Use the model compiled into the binary (#237) rather than a file on disk.
    /// True for the zero-config default; false when `--embedding-model` points at
    /// a custom ONNX file.
    #[allow(dead_code)]
    pub bundled: bool,
}

impl EmbeddingConfig {
    #[allow(dead_code)]
    pub fn with_model_path(path: PathBuf) -> Self {
        EmbeddingConfig {
            enabled: true,
            model_path: path,
            bundled: false,
        }
    }

    /// Default model path in ~/.perseus-vault/models/
    pub fn default_path() -> PathBuf {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home)
            .join(".perseus-vault")
            .join("models")
            .join("all-MiniLM-L6-v2")
            .join("model.onnx")
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            // #237: with the bundled model compiled in (default feature), local
            // dense embeddings are available with zero config — so enable them by
            // default. Without the feature (lite build) they stay off, falling
            // back to a remote endpoint if one is configured.
            enabled: cfg!(feature = "bundled-embeddings"),
            bundled: cfg!(feature = "bundled-embeddings"),
            model_path: Self::default_path(),
        }
    }
}

// ─── Model Download ─────────────────────────────────────────────────────

/// Download the all-MiniLM-L6-v2 ONNX model from HuggingFace if not already cached.
#[allow(dead_code)]
pub fn ensure_model(config: &EmbeddingConfig) -> Result<(), String> {
    let model_dir = config
        .model_path
        .parent()
        .ok_or_else(|| "invalid model path".to_string())?;

    std::fs::create_dir_all(model_dir)
        .map_err(|e| format!("failed to create model directory: {}", e))?;

    if !config.model_path.exists() {
        eprintln!(
            "perseus-vault: downloading embedding model to {} ...",
            config.model_path.display()
        );
        download_file(
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
            &config.model_path,
        )?;
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!(
            "perseus-vault: downloading tokenizer to {} ...",
            tokenizer_path.display()
        );
        download_file(
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
            &tokenizer_path,
        )?;
    }

    Ok(())
}

#[allow(dead_code)]
fn download_file(url: &str, dest: &PathBuf) -> Result<(), String> {
    let response = ureq::get(url)
        .timeout(std::time::Duration::from_secs(600))
        .call()
        .map_err(|e| format!("download failed for {}: {}", url, e))?;

    let total = response
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok());

    let mut reader = response.into_reader();
    let mut file =
        std::fs::File::create(dest).map_err(|e| format!("failed to create file: {}", e))?;

    let mut buf = [0u8; 65536];
    let mut downloaded: u64 = 0;
    loop {
        let n =
            std::io::Read::read(&mut reader, &mut buf).map_err(|e| format!("read error: {}", e))?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])
            .map_err(|e| format!("write error: {}", e))?;
        downloaded += n as u64;
        if let Some(total) = total {
            if downloaded % (1024 * 1024) < 65536 || downloaded == total {
                eprint!(
                    "\r  {:.1}% ({:.1} MB / {:.1} MB)",
                    (downloaded as f64 / total as f64) * 100.0,
                    downloaded as f64 / (1024.0 * 1024.0),
                    total as f64 / (1024.0 * 1024.0)
                );
            }
        }
    }
    if total.is_some() {
        eprintln!();
    }
    Ok(())
}

// ─── Embedding Generation ───────────────────────────────────────────────

/// Generate a 384-dimensional embedding vector for the given text.
///
/// Tries backends in order:
///   1. Native ort+tokenizers (if feature "bundled-embeddings" was enabled at build time)
///   2. Python onnxruntime (if `python3` is on PATH and `onnxruntime` is installed)
///   3. Returns an error suggesting either option
#[allow(dead_code)]
pub fn generate_embedding(
    config: &EmbeddingConfig,
    text: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Bundled model is compiled into the binary — no download/setup needed.
    if !config.bundled {
        ensure_model(config).map_err(|e| format!("model setup failed: {}", e))?;
    }

    // Try native ort backend first
    #[cfg(feature = "bundled-embeddings")]
    {
        match generate_with_ort(config, text) {
            Ok(vec) => return Ok(vec),
            Err(e) => eprintln!(
                "perseus-vault: native embedding failed ({}), trying Python fallback...",
                e
            ),
        }
    }

    // Try Python onnxruntime fallback
    match generate_with_python(config, text) {
        Ok(vec) => return Ok(vec),
        Err(e) => eprintln!("perseus-vault: Python embedding fallback failed ({})", e),
    }

    Err("No embedding backend available. Options:\n\
         - Rebuild with: cargo build --release --features bundled-embeddings\n\
         - Install Python + onnxruntime: pip install onnxruntime\n\
         - Use Ollama: perseus_vault serve --llm-endpoint http://localhost:11434"
        .into())
}

// ─── Native ort backend (feature = "bundled-embeddings") ─────────────────

/// A loaded ONNX model + its tokenizer, cached for the process lifetime so the
/// ~80MB graph and tokenizer.json are not re-loaded on every embedding call
/// (#208). `Session::run` takes `&mut self` (rc.12), so the session is behind a
/// Mutex; the tokenizer's `encode` takes `&self` and is shared directly.
#[cfg(feature = "bundled-embeddings")]
struct OrtModel {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
}

/// Return the process-cached model for `config.model_path`, loading + caching it
/// on first use. Keyed by path so a reconfigured model loads fresh rather than
/// reusing a stale one; entries live for the process lifetime. The cold load
/// runs under the map lock so two cold callers don't both build the graph. (#208)
#[cfg(feature = "bundled-embeddings")]
fn cached_ort_model(
    config: &EmbeddingConfig,
) -> Result<std::sync::Arc<OrtModel>, Box<dyn std::error::Error>> {
    use ort::session::Session;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<OrtModel>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache
        .lock()
        .map_err(|_| "embedding model cache mutex poisoned")?;
    // Key the bundled model by a sentinel so it shares one cached session
    // regardless of the (unused) default model_path.
    let key = if config.bundled {
        PathBuf::from("<bundled:all-MiniLM-L6-v2>")
    } else {
        config.model_path.clone()
    };
    if let Some(model) = map.get(&key) {
        return Ok(Arc::clone(model));
    }

    let (session, tokenizer) = if config.bundled {
        // #237: load the compiled-in model + tokenizer straight from memory — no
        // file on disk, no network.
        // #310: pin a single intra-op thread + deterministic compute so the
        // embedding is bit-reproducible run-to-run. Multi-threaded ORT reduces
        // in nondeterministic order, producing tiny FP differences that flip
        // near-tied cosine ranks (≈0.3% of LongMemEval at scale). The model is
        // tiny (MiniLM-L6, short inputs) and results are LRU-cached, so the
        // single-thread cost is negligible.
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_deterministic_compute(true)?
            .commit_from_memory(BUNDLED_MODEL)?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(BUNDLED_TOKENIZER)
            .map_err(|e| format!("failed to load bundled tokenizer: {}", e))?;
        (session, tokenizer)
    } else {
        let model_dir = config
            .model_path
            .parent()
            .ok_or("model_path must have a parent directory")?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_deterministic_compute(true)?
            .commit_from_file(&config.model_path)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("failed to load tokenizer: {}", e))?;
        (session, tokenizer)
    };
    let model = Arc::new(OrtModel {
        session: Mutex::new(session),
        tokenizer,
    });
    map.insert(key, Arc::clone(&model));
    Ok(model)
}

#[cfg(feature = "bundled-embeddings")]
fn generate_with_ort(
    config: &EmbeddingConfig,
    text: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // #208: reuse the cached session + tokenizer instead of rebuilding (loading
    // the ~80MB ONNX graph + parsing tokenizer.json) on every call.
    let model = cached_ort_model(config)?;

    let encoding = model
        .tokenizer
        .encode(text, true)
        .map_err(|e| format!("tokenization failed: {}", e))?;

    let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();

    // BERT models (incl. all-MiniLM-L6-v2) take three inputs; token_type_ids is
    // all-zeros for a single sequence. The quantized export *requires* it (the
    // graph has a token_type_embeddings Gather), so omitting it fails at runtime —
    // pass it explicitly. (#237)
    let type_ids: Vec<i64> = vec![0i64; token_ids.len()];

    // Use ndarray for tensor creation (ort 2.x uses ndarray). Bind the
    // batch-axis arrays to locals so they outlive the borrowing TensorRefs
    // through session.run (rc.12: from_array_view borrows the array). (#212)
    let input_2d = ndarray::Array1::from_vec(token_ids.clone()).insert_axis(ndarray::Axis(0));
    let mask_2d = ndarray::Array1::from_vec(attention_mask.clone()).insert_axis(ndarray::Axis(0));
    let types_2d = ndarray::Array1::from_vec(type_ids).insert_axis(ndarray::Axis(0));

    let input_tensor = ort::value::TensorRef::from_array_view(&input_2d)?;
    let mask_tensor = ort::value::TensorRef::from_array_view(&mask_2d)?;
    let types_tensor = ort::value::TensorRef::from_array_view(&types_2d)?;

    // ort 2.0.0-rc.12: `inputs!` yields the inputs directly (no longer a Result),
    // so there is no inner `?`. Lock the cached session for the run; the guard is
    // held through output extraction below (outputs borrow the session). (#212/#208)
    let mut session = model
        .session
        .lock()
        .map_err(|_| "embedding session mutex poisoned")?;
    let outputs = session.run(ort::inputs![
        "input_ids" => input_tensor,
        "attention_mask" => mask_tensor,
        "token_type_ids" => types_tensor,
    ])?;

    // Extract last_hidden_state and mean pool. rc.12 replaced `extract_tensor`
    // (which no longer exists on a dynamic Value) with `try_extract_tensor`,
    // returning (&Shape, &[T]) — a flat row-major slice. The tensor is
    // [batch=1, seq_len, dim], so element [0, t, d] is at offset t*dim + d. (#212)
    let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
    let seq_len = shape[1] as usize;
    let dim = shape[2] as usize;

    let mut pooled = vec![0.0f32; dim];
    let mut active = 0usize;
    for t in 0..seq_len {
        if t < attention_mask.len() && attention_mask[t] == 1 {
            let row = t * dim;
            for d in 0..dim {
                pooled[d] += data[row + d];
            }
            active += 1;
        }
    }
    if active > 0 {
        let n = active as f32;
        for v in pooled.iter_mut() {
            *v /= n;
        }
    }

    // L2 normalize
    let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in pooled.iter_mut() {
            *v /= norm;
        }
    }

    Ok(pooled)
}

// ─── Python onnxruntime fallback ─────────────────────────────────────────

/// Generate embeddings using a Python helper that calls onnxruntime.
#[allow(dead_code)]
fn generate_with_python(
    config: &EmbeddingConfig,
    text: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let model_dir = config
        .model_path
        .parent()
        .ok_or_else(|| "invalid model path".to_string())?;
    let tokenizer_path = model_dir.join("tokenizer.json");

    let model_str = config.model_path.to_string_lossy();
    let tokenizer_str = tokenizer_path.to_string_lossy();

    // Pass the tokenizer path, model path and text as argv, NOT interpolated into
    // the script source. The previous version only escaped `\` and `'`, so a
    // newline (or other control char) in the text broke out of the single-quoted
    // Python string literal — a code-injection / DoS hazard since the text is
    // agent/user-controlled. argv values are never parsed as Python code.
    let script = r#"
import sys, json, numpy as np
try:
    import onnxruntime as ort
except ImportError:
    print(json.dumps({"error": "onnxruntime not installed. Run: pip install onnxruntime"}))
    sys.exit(1)

try:
    from tokenizers import Tokenizer
    tokenizer_path, model_path, text = sys.argv[1], sys.argv[2], sys.argv[3]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

    session = ort.InferenceSession(model_path)
    outputs = session.run(None, {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    })
    hidden = outputs[0]  # [1, seq_len, 384]

    # Mean pooling with attention mask
    mask = attention_mask[0, :, None]  # [seq_len, 1]
    pooled = (hidden[0] * mask).sum(axis=0) / mask.sum()
    # L2 normalize
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm

    print(json.dumps({"embedding": pooled.tolist()}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
"#;

    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(tokenizer_str.as_ref())
        .arg(model_str.as_ref())
        .arg(text)
        .output()
        .map_err(|e| {
            format!(
                "failed to run python3: {}. Install Python 3 and onnxruntime.",
                e
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("python3 embedding failed: {}", stderr).into());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: serde_json::Value = serde_json::from_str(&stdout)
        .map_err(|e| format!("failed to parse python output: {} — raw: {}", e, stdout))?;

    if let Some(err) = result.get("error") {
        return Err(format!("python embedding error: {}", err).into());
    }

    let embedding: Vec<f32> = result["embedding"]
        .as_array()
        .ok_or("missing embedding in python output")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();

    Ok(embedding)
}

// ─── End-to-end test for the bundled model (#237) ────────────────────────
// Runs only in a bundled-embeddings build: it loads the compiled-in quantized
// model from memory and runs real inference, validating the in-memory load path
// and the model's (input_ids, attention_mask) -> last_hidden_state signature —
// which a plain `cargo build` would not exercise.
#[cfg(all(test, feature = "bundled-embeddings"))]
mod bundled_tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn bundled_model_embeds_and_is_semantic() {
        let cfg = EmbeddingConfig::default();
        assert!(
            cfg.enabled && cfg.bundled,
            "bundled config should be on by default"
        );

        let v = generate_embedding(&cfg, "hello world").expect("bundled embedding works");
        assert_eq!(v.len(), 384, "all-MiniLM-L6-v2 is 384-dim");

        // Output is L2-normalized → unit norm.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "expected unit vector, norm={norm}"
        );

        // Semantically related texts rank above an unrelated one (sanity that the
        // quantized model produces meaningful embeddings, not noise).
        let a = generate_embedding(&cfg, "cats and dogs").unwrap();
        let b = generate_embedding(&cfg, "kittens and puppies").unwrap();
        let c = generate_embedding(&cfg, "quarterly financial report").unwrap();
        assert!(
            cosine(&a, &b) > cosine(&a, &c),
            "related texts should be more similar than unrelated ones"
        );
    }
}
