use std::sync::Arc;

use metrics_exporter_prometheus::PrometheusHandle;

use crate::services::{EmbeddingProvider, LlmProvider, SemanticCache};

/// Shared application state, cloned (cheaply via Arc) for every request.
#[derive(Clone)]
pub struct AppState {
    /// LLM HTTP client — text generation.
    pub ollama: Arc<dyn LlmProvider>,
    /// Embedding service - either local rust-bert or Ollama API.
    pub embedder: Arc<dyn EmbeddingProvider>,
    /// Moka-backed semantic cache.
    pub cache: Arc<SemanticCache>,
    /// Prometheus metrics renderer.
    pub metrics_handle: PrometheusHandle,
}

