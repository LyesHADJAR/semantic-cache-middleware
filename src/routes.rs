use axum::Router;
use axum::routing::{get, post};

use crate::handlers;
use crate::state::AppState;

/// Build the application router with all routes and shared state.
pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::root))
        .route("/generate", post(handlers::generate))
        .with_state(state)
}
