use std::sync::Arc;
use tracing::info;

use semantic_cache_middleware::config::AppConfig;
use semantic_cache_middleware::services::{
    EmbeddingProvider, LocalEmbeddingService, LlmProvider, OllamaEmbeddingService, OllamaService,
    SemanticCache,
};
use semantic_cache_middleware::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // 1. Load configuration from env vars (or defaults).
    let config = AppConfig::from_env();

    // 2. Install the Prometheus metrics recorder (must happen before any metrics are emitted).
    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    // 3. Build the Ollama HTTP client (text generation).
    let ollama: Arc<dyn LlmProvider> = Arc::new(OllamaService::new(&config));

    // 4. Start the configured embedding service.
    info!("Initializing system components...");
    let embedder: Arc<dyn EmbeddingProvider> = match config.embedding_provider.as_str() {
        "local" => Arc::new(LocalEmbeddingService::init()?),
        "ollama" => Arc::new(OllamaEmbeddingService::new(&config)),
        _ => unreachable!(),
    };

    // 5. Initialize the semantic cache.
    let cache = SemanticCache::new(config.similarity_threshold);

    // 6. Assemble the shared application state.
    let app_state = AppState {
        ollama,
        embedder,
        cache,
        metrics_handle,
    };

    // 7. Build the router.
    let app = semantic_cache_middleware::routes::app(app_state);

    // 8. Start the server.
    let listener = tokio::net::TcpListener::bind(&config.listen_addr)
        .await
        .expect("failed to bind address");

    info!("Server running at http://{}", config.listen_addr);
    axum::serve(listener, app).await?;

    Ok(())
}
