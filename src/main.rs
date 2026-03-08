mod config;
mod error;
mod handlers;
mod models;
mod routes;
mod services;
mod state;

use std::sync::Arc;

use config::AppConfig;
use services::OllamaService;
use state::AppState;

use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Load configuration from env vars (or defaults).
    let config = AppConfig::from_env();

    // 2. Build the Ollama HTTP client (text generation).
    let ollama = Arc::new(OllamaService::new(&config));

    // 3. Load the sentence-embedding model on a blocking thread.
    //    This downloads the model on first run and is CPU-heavy,
    //    so we must not run it on the async executor.
    println!("Loading embedding model (this may take a while on first run)...");
    let embedder = tokio::task::spawn_blocking(|| {
        SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
            .create_model()
    })
    .await??; // first ? unwraps the JoinError, second ? unwraps the rust-bert error
    println!("Embedding model loaded.");

    // 4. Assemble the shared application state.
    let app_state = AppState {
        ollama,
        embedder: Arc::new(tokio::sync::Mutex::new(embedder)),
    };

    // 5. Build the router.
    let app = routes::app(app_state);

    // 6. Start the server.
    let listener = tokio::net::TcpListener::bind(&config.listen_addr)
        .await
        .expect("failed to bind address");

    println!("Server running at http://{}", config.listen_addr);
    axum::serve(listener, app).await?;

    Ok(())
}
