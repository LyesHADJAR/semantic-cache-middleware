use crate::config::AppConfig;
use crate::error::AppError;

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
        Self {
            client: reqwest::Client::new(),
            base_url: config.ollama_base_url.clone(),
            model: config.ollama_model.clone(),
        }
    }

    /// Send a prompt to Ollama and return the concatenated response text.
    pub async fn generate(&self, prompt: &str) -> Result<String, AppError> {
        let url = format!("{}/api/generate", self.base_url);

        let json_body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
        });

        let body = self
            .client
            .post(&url)
            .json(&json_body)
            .send()
            .await?
            .text()
            .await
            .map_err(|e| AppError::ResponseParse(e.to_string()))?;

        let output = body
            .lines()
            .filter_map(|line| {
                serde_json::from_str::<serde_json::Value>(line)
                    .ok()
                    .and_then(|json| json.get("response")?.as_str().map(String::from))
            })
            .collect::<String>();

        Ok(output)
    }
}
