use axum::Router;
use axum::routing::{get, post};
use std::time::Duration;
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer, timeout::TimeoutLayer};

use crate::handlers;
use crate::state::AppState;

/// Build the application router with all routes and shared state.
pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/generate", post(handlers::generate))
        .route("/metrics", get(handlers::metrics_handler))
        .with_state(state)
        // Global timeout of 120s to prevent slow client attacks or hung models
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(120),
        ))
        // 2 MB strict body limit to prevent memory-based DoS
        .layer(RequestBodyLimitLayer::new(2 * 1024 * 1024))
        // Basic CORS policy
        .layer(CorsLayer::permissive())
}
