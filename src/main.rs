mod config;
mod error;
mod handlers;
mod models;
mod routes;
mod services;
mod state;

use std::sync::Arc;

use config::AppConfig;
use services::{EmbeddingService, OllamaService, SemanticCache};
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Load configuration from env vars (or defaults).
    let config = AppConfig::from_env();

    // 2. Build the Ollama HTTP client (text generation).
    let ollama = Arc::new(OllamaService::new(&config));

    // 3. Start the embedding worker service (Actor).
    let embedder = EmbeddingService::new();

    // 4. Initialize the semantic cache.
    let cache = Arc::new(SemanticCache::new(config.similarity_threshold));

    // 5. Assemble the shared application state.
    let app_state = AppState {
        ollama,
        embedder,
        cache,
    };

    // 6. Build the router.
    let app = routes::app(app_state);

    // 7. Start the server.
    let listener = tokio::net::TcpListener::bind(&config.listen_addr)
        .await
        .expect("failed to bind address");

    println!("Server running at http://{}", config.listen_addr);
    axum::serve(listener, app).await?;

    Ok(())
}
