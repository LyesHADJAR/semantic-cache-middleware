use metrics::{counter, histogram};
use std::time::Instant;

/// Metric key constants to avoid string typos across the codebase.
const CACHE_HITS_EXACT: &str = "cache_hits_exact_total";
const CACHE_HITS_SEMANTIC: &str = "cache_hits_semantic_total";
const CACHE_MISSES: &str = "cache_misses_total";
const EMBEDDING_LATENCY: &str = "embedding_latency_seconds";
const GENERATION_LATENCY: &str = "generation_latency_seconds";
const REQUEST_LATENCY: &str = "request_latency_seconds";

/// Record an exact cache hit.
pub fn record_exact_hit() {
    counter!(CACHE_HITS_EXACT).increment(1);
}

/// Record a semantic (HNSW) cache hit.
pub fn record_semantic_hit() {
    counter!(CACHE_HITS_SEMANTIC).increment(1);
}

/// Record a cache miss (full generation required).
pub fn record_miss() {
    counter!(CACHE_MISSES).increment(1);
}

/// Record how long embedding computation took.
pub fn record_embedding_latency(start: Instant) {
    histogram!(EMBEDDING_LATENCY).record(start.elapsed().as_secs_f64());
}

/// Record how long LLM text generation took.
pub fn record_generation_latency(start: Instant) {
    histogram!(GENERATION_LATENCY).record(start.elapsed().as_secs_f64());
}

/// Record total request latency for the `/generate` endpoint.
pub fn record_request_latency(start: Instant) {
    histogram!(REQUEST_LATENCY).record(start.elapsed().as_secs_f64());
}
