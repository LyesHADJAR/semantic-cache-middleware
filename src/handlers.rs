use axum::Json;
use axum::extract::State;
use tracing::info;

use crate::error::AppError;
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
    let prompt = payload.prompt;

    // 1. Fast exact-match check
    if let Some(entry) = app_state.cache.get_exact(&prompt) {
        info!("Exact cache hit for prompt: {}", prompt);
        return Ok(Json(GenerateResponse {
            response_text: entry.response_text,
            embedding: entry.embedding,
        }));
    }

    // 2. Exact match missed, let's compute the embedding
    let query_embedding = app_state.embedder.encode(prompt.clone()).await?;

    // 3. Semantic search
    if let Some(entry) = app_state.cache.search_semantic(&query_embedding) {
        info!("Semantic cache hit for prompt: {}", prompt);
        return Ok(Json(GenerateResponse {
            response_text: entry.response_text,
            embedding: query_embedding,
        }));
    }

    info!("Cache miss for prompt: {}", prompt);

    // 4. Cache miss: generate text via Ollama
    let generated_text = app_state.ollama.generate(&prompt).await?;

    // 5. Insert into cache
    app_state.cache.insert(
        prompt.clone(),
        query_embedding.clone(),
        generated_text.clone(),
    );

    Ok(Json(GenerateResponse {
        response_text: generated_text,
        embedding: query_embedding,
    }))
}
