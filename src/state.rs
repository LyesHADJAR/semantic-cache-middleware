use std::sync::Arc;

use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use tokio::sync::Mutex;

use crate::services::OllamaService;

/// Shared application state, cloned (cheaply via Arc) for every request.
#[derive(Clone)]
pub struct AppState {
    /// Ollama HTTP client — text generation.
    pub ollama: Arc<OllamaService>,
    /// Sentence-embedding model — lives behind a Mutex because it's not Send+Sync
    /// friendly and requires exclusive (&mut) access.
    pub embedder: Arc<Mutex<SentenceEmbeddingsModel>>,
}