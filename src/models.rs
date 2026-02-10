use serde::Deserialize;

/// Incoming request body for the `/generate` endpoint.
#[derive(Debug, Deserialize)]
pub struct PromptRequest {
    pub prompt: String,
}
