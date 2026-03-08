use serde::Deserialize; 
use serde::Serialize;

/// Incoming request body for the `/generate` endpoint.
#[derive(Debug, Deserialize)]
pub struct PromptRequest {
    pub prompt: String,
}

/// A single chunk from the Ollama streaming NDJSON response.
#[derive(Debug, Deserialize)]
pub struct OllamaChunk {
    /// The partial text generated so far.
    pub response: String,
    /// Whether this is the final chunk.
    #[allow(dead_code)]
    pub done: bool,
}

#[derive(Serialize)]
pub struct GenerateResponse {
    pub response_text: String,
    pub embedding: Vec<Vec<f32>>,
}
