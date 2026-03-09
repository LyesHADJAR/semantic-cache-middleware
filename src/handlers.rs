use axum::Json;
use axum::extract::State;
use std::time::Instant;
use tracing::info;

use crate::error::AppError;
use crate::metrics as app_metrics;
use crate::models::{GenerateResponse, PromptRequest};
use crate::state::AppState;

/// Health-check / hello-world handler.
pub async fn root() -> &'static str {
    "Hello, World!"
}

/// Accept a prompt, fetch embedding, check cache, forward to Ollama if miss, and return.
pub async fn generate(
    State(app_state): State<AppState>,
    Json(payload): Json<PromptRequest>,
) -> Result<Json<GenerateResponse>, AppError> {
    let request_start = Instant::now();
    let prompt = payload.prompt.trim().to_string();

    if prompt.is_empty() {
        return Err(AppError::ValidationError("Prompt cannot be empty".into()));
    }

    if prompt.len() > 100_000 {
        return Err(AppError::ValidationError(
            "Prompt is too long (max 100,000 characters)".into(),
        ));
    }

    // 1. Fast exact-match check.
    if let Some(entry) = app_state.cache.get_exact(&prompt) {
        info!("Exact cache hit for prompt: {}", prompt);
        app_metrics::record_exact_hit();
        app_metrics::record_request_latency(request_start);
        return Ok(Json(GenerateResponse {
            response_text: entry.response_text,
            embedding: entry.embedding,
        }));
    }

    // 2. Exact match missed — compute embedding.
    let embed_start = Instant::now();
    let query_embedding = app_state.embedder.encode(prompt.clone()).await?;
    app_metrics::record_embedding_latency(embed_start);

    // 3. Semantic search via HNSW.
    if let Some(entry) = app_state.cache.search_semantic(&query_embedding) {
        info!("Semantic cache hit for prompt: {}", prompt);
        app_metrics::record_semantic_hit();
        app_metrics::record_request_latency(request_start);
        return Ok(Json(GenerateResponse {
            response_text: entry.response_text,
            embedding: query_embedding,
        }));
    }

    info!("Cache miss for prompt: {}", prompt);
    app_metrics::record_miss();

    // 4. Cache miss: generate text via Ollama.
    let gen_start = Instant::now();
    let generated_text = app_state.ollama.generate(&prompt).await?;
    app_metrics::record_generation_latency(gen_start);

    // 5. Insert into cache.
    app_state.cache.insert(
        prompt.clone(),
        query_embedding.clone(),
        generated_text.clone(),
    );

    app_metrics::record_request_latency(request_start);
    Ok(Json(GenerateResponse {
        response_text: generated_text,
        embedding: query_embedding,
    }))
}

/// Serve Prometheus metrics as plain-text scrape output.
pub async fn metrics_handler(
    State(app_state): State<AppState>,
) -> String {
    app_state.metrics_handle.render()
}
