use axum::extract::State;
use axum::Json;

use crate::error::AppError;
use crate::models::PromptRequest;
use crate::services::OllamaService;

/// Health-check / hello-world handler.
pub async fn root() -> &'static str {
    "Hello, World!"
}

/// Accept a prompt, forward it to Ollama, and return the generated text.
pub async fn generate(
    State(ollama): State<OllamaService>,
    Json(payload): Json<PromptRequest>,
) -> Result<String, AppError> {
    ollama.generate(&payload.prompt).await
}
