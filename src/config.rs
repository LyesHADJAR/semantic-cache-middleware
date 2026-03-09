/// Application configuration, loaded once at startup.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Address the server binds to (e.g. "127.0.0.1:3000").
    pub listen_addr: String,
    /// Base URL of the Ollama API.
    pub ollama_base_url: String,
    /// Default model to use for generation.
    pub ollama_model: String,
    /// Cosine similarity threshold for cache hits (0.0 to 1.0).
    pub similarity_threshold: f32,
    /// Which embedding provider to use ("local" or "ollama").
    pub embedding_provider: String,
    /// Default model to use for Ollama embeddings.
    pub ollama_embedding_model: String,
}

impl AppConfig {
    /// Build the config from environment variables, falling back to sensible
    /// defaults for local development.
    pub fn from_env() -> Self {
        let config = Self {
            listen_addr: std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "127.0.0.1:3000".into()),
            ollama_base_url: std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".into()),
            ollama_model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".into()),
            similarity_threshold: std::env::var("SIMILARITY_THRESHOLD")
                .unwrap_or_else(|_| "0.85".into())
                .parse()
                .unwrap_or(0.85),
            embedding_provider: std::env::var("EMBEDDING_PROVIDER")
                .unwrap_or_else(|_| "local".into()),
            ollama_embedding_model: std::env::var("OLLAMA_EMBEDDING_MODEL")
                .unwrap_or_else(|_| "nomic-embed-text".into()),
        };
        config.validate();
        config
    }

    /// Validates the configuration and panics immediately if invalid.
    fn validate(&self) {
        if self.embedding_provider != "local" && self.embedding_provider != "ollama" {
            panic!(
                "Invalid EMBEDDING_PROVIDER '{}'. Must be 'local' or 'ollama'",
                self.embedding_provider
            );
        }
        if self.ollama_base_url.is_empty() {
            panic!("OLLAMA_BASE_URL cannot be empty");
        }
    }
}
