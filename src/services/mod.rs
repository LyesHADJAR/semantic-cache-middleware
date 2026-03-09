pub mod cache;
pub mod embedding;
pub mod ollama;

pub use cache::SemanticCache;
pub use embedding::{EmbeddingProvider, LocalEmbeddingService, OllamaEmbeddingService};
pub use ollama::{LlmProvider, OllamaService};

#[cfg(test)]
pub use embedding::MockEmbeddingProvider;
#[cfg(test)]
pub use ollama::MockLlmProvider;
