use std::sync::Arc;

use axum::http::StatusCode;
use axum_test::TestServer;
use serde_json::json;

use crate::routes;
use crate::services::{MockEmbeddingProvider, MockLlmProvider, SemanticCache};
use crate::state::AppState;

/// Build a test `AppState` with mocked services.
fn build_test_state(mock_embedder: MockEmbeddingProvider, mock_llm: MockLlmProvider) -> AppState {
    let cache = SemanticCache::new(0.85);
    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();

    AppState {
        ollama: Arc::new(mock_llm),
        embedder: Arc::new(mock_embedder),
        cache,
        metrics_handle,
    }
}

#[tokio::test]
async fn health_check_returns_ok() {
    let state = build_test_state(MockEmbeddingProvider::new(), MockLlmProvider::new());
    let server = TestServer::new(routes::app(state));

    let response = server.get("/").await;
    response.assert_status_ok();
    response.assert_text("Hello, World!");
}

#[tokio::test]
async fn generate_rejects_empty_prompt() {
    let state = build_test_state(MockEmbeddingProvider::new(), MockLlmProvider::new());
    let server = TestServer::new(routes::app(state));

    let response = server
        .post("/generate")
        .json(&json!({ "prompt": "   " }))
        .await;

    response.assert_status(StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn generate_cache_miss_calls_llm_and_returns() {
    let mut mock_embedder = MockEmbeddingProvider::new();
    let mut mock_llm = MockLlmProvider::new();

    let embedding = vec![0.1, 0.2, 0.3];
    let embedding_clone = embedding.clone();

    mock_embedder
        .expect_encode()
        .returning(move |_| Ok(embedding_clone.clone()));

    mock_llm
        .expect_generate()
        .returning(|_| Ok("This is a response from the LLM.".to_string()));

    let state = build_test_state(mock_embedder, mock_llm);
    let server = TestServer::new(routes::app(state));

    let response = server
        .post("/generate")
        .json(&json!({ "prompt": "Test prompt" }))
        .await;

    response.assert_status_ok();
    let body: serde_json::Value = response.json();
    assert_eq!(body["response_text"], "This is a response from the LLM.");
    assert!(body["embedding"].is_array());
}

#[tokio::test]
async fn generate_exact_cache_hit_does_not_call_llm() {
    let mut mock_embedder = MockEmbeddingProvider::new();
    let mut mock_llm = MockLlmProvider::new();

    let embedding = vec![0.1, 0.2, 0.3];
    let embedding_clone = embedding.clone();

    // Embedder + LLM should only be called once (on the first, cache-miss request).
    mock_embedder
        .expect_encode()
        .times(1)
        .returning(move |_| Ok(embedding_clone.clone()));

    mock_llm
        .expect_generate()
        .times(1)
        .returning(|_| Ok("Generated response".to_string()));

    let state = build_test_state(mock_embedder, mock_llm);
    let server = TestServer::new(routes::app(state));

    // First request: cache miss.
    server
        .post("/generate")
        .json(&json!({ "prompt": "Same prompt" }))
        .await
        .assert_status_ok();

    // Second request: exact hit — neither embedder nor LLM should be called again.
    let response = server
        .post("/generate")
        .json(&json!({ "prompt": "Same prompt" }))
        .await;

    response.assert_status_ok();
    let body: serde_json::Value = response.json();
    assert_eq!(body["response_text"], "Generated response");
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_output() {
    let state = build_test_state(MockEmbeddingProvider::new(), MockLlmProvider::new());
    let server = TestServer::new(routes::app(state));

    let response = server.get("/metrics").await;
    response.assert_status_ok();
}

#[tokio::test]
async fn cache_insert_and_exact_retrieval() {
    let cache = SemanticCache::new(0.85);
    let prompt = "What is Rust?".to_string();
    let embedding = vec![0.5, 0.6, 0.7];
    let response_text = "Rust is a systems language.".to_string();

    cache.insert(prompt.clone(), embedding.clone(), response_text.clone());

    let entry = cache.get_exact(&prompt).expect("exact hit should exist");
    assert_eq!(entry.response_text, response_text);
    assert_eq!(entry.embedding, embedding);
}

#[tokio::test]
async fn cache_semantic_search_hit() {
    let cache = SemanticCache::new(0.90);
    let embedding = vec![1.0, 0.0, 0.0];

    cache.insert(
        "original prompt".to_string(),
        embedding.clone(),
        "cached response".to_string(),
    );

    let entry = cache
        .search_semantic(&embedding)
        .expect("semantic hit should exist");
    assert_eq!(entry.response_text, "cached response");
}

#[tokio::test]
async fn cache_semantic_search_miss_for_distant_vector() {
    let cache = SemanticCache::new(0.99);
    let original = vec![1.0, 0.0, 0.0];
    let distant = vec![0.0, 0.0, 1.0];

    cache.insert("prompt".to_string(), original, "response".to_string());

    // A completely orthogonal vector should be a miss at threshold 0.99.
    assert!(cache.search_semantic(&distant).is_none());
}
