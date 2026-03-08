use axum::extract::State;
use axum::Json;

use crate::error::AppError;
use crate::models::{GenerateResponse, PromptRequest};
use crate::state::AppState;

/// Health-check / hello-world handler.
pub async fn root() -> &'static str {
    "Hello, World!"
}

/// Accept a prompt, embed it, forward it to Ollama, and return both.
pub async fn generate(
    State(app_state): State<AppState>,
    Json(payload): Json<PromptRequest>,
) -> Result<Json<GenerateResponse>, AppError> {
    // 1. Generate text via Ollama (async HTTP call).
    let generated_text = app_state.ollama.generate(&payload.prompt).await?;

    // 2. Compute the embedding via rust-bert.
    //    The model needs &mut self, so we lock the mutex.
    //    encode() is CPU-heavy and blocking, so we run it on a
    //    blocking thread to avoid stalling the Tokio event loop.
    let embedder = app_state.embedder.clone();
    let prompt_clone = payload.prompt.clone();
    let embedded_prompt = tokio::task::spawn_blocking(move || {
        let model = embedder.blocking_lock();
        model.encode(&[prompt_clone])
    })
    .await
    .map_err(|e| AppError::Embedding(format!("task join error: {e}")))?  // JoinError
    .map_err(|e| AppError::Embedding(format!("encode error: {e}")))?;   // rust-bert error

    let response = GenerateResponse {
        response_text: generated_text,
        embedding: embedded_prompt,
    };
    Ok(Json(response))
}
