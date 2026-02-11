use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

use crate::handlers;
use crate::services::OllamaService;

/// Build the application router with all routes and shared state.
pub fn app(ollama: Arc<OllamaService>) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/generate", post(handlers::generate))
        .with_state(ollama)
}
