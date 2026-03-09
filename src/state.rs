use std::sync::Arc;

use crate::services::{EmbeddingService, LlmProvider, SemanticCache};

/// Shared application state, cloned (cheaply via Arc) for every request.
#[derive(Clone)]
pub struct AppState {
    /// LLM HTTP client — text generation.
    pub ollama: Arc<dyn LlmProvider>,
    /// Embedding service - communicates with the native embedding worker thread.
    pub embedder: EmbeddingService,
    /// Moka-backed semantic cache
    pub cache: Arc<SemanticCache>,
}
