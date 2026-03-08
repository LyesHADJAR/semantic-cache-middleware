use std::sync::Arc;

use crate::services::{EmbeddingService, OllamaService, SemanticCache};

/// Shared application state, cloned (cheaply via Arc) for every request.
#[derive(Clone)]
pub struct AppState {
    /// Ollama HTTP client — text generation.
    pub ollama: Arc<OllamaService>,
    /// Embedding service - communicates with the native embedding worker thread.
    pub embedder: EmbeddingService,
    /// DashMap-backed semantic cache
    pub cache: Arc<SemanticCache>,
}
