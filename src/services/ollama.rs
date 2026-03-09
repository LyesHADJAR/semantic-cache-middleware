use std::time::Duration;

use crate::config::AppConfig;
use crate::error::AppError;
use crate::models::OllamaChunk;

/// Default overall request timeout (LLM generation can be slow).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// TCP connect timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    async fn generate(&self, prompt: &str) -> Result<String, AppError>;
}

/// Thin wrapper around the Ollama REST API.
#[derive(Debug, Clone)]
pub struct OllamaService {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaService {
    /// Create a new service instance from the application config.
    pub fn new(config: &AppConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            base_url: config.ollama_base_url.clone(),
            model: config.ollama_model.clone(),
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for OllamaService {
    /// Send a prompt to Ollama and return the concatenated response text.
    async fn generate(&self, prompt: &str) -> Result<String, AppError> {
        let url = format!("{}/api/generate", self.base_url);

        let json_body = serde_json::json!({
            "model": &self.model,
            "prompt": prompt,
        });

        let body = self
            .client
            .post(&url)
            .json(&json_body)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await
            .map_err(|e| AppError::ResponseParse(e.to_string()))?;

        let mut output = String::new();
        for line in body.lines() {
            let chunk: OllamaChunk = serde_json::from_str(line)
                .map_err(|e| AppError::ResponseParse(format!("malformed chunk: {e}")))?;
            output.push_str(&chunk.response);
        }

        if output.is_empty() {
            return Err(AppError::EmptyResponse);
        }

        Ok(output)
    }
}
