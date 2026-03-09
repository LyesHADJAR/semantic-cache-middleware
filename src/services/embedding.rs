use crate::error::AppError;
use crate::config::AppConfig;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use std::time::Duration;
use std::thread;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};
use serde::Deserialize;

#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn encode(&self, prompt: String) -> Result<Vec<f32>, AppError>;
}

/// Message sent to the local embedding worker thread.
type EmbeddingMessage = (String, oneshot::Sender<Result<Vec<f32>, AppError>>);

/// Local embedding service using `rust-bert` on a background thread.
#[derive(Clone)]
pub struct LocalEmbeddingService {
    sender: mpsc::Sender<EmbeddingMessage>,
}

impl LocalEmbeddingService {
    pub fn init() -> Result<Self, AppError> {
        let (init_tx, init_rx) = std::sync::mpsc::sync_channel(1);
        let (tx, mut rx) = mpsc::channel::<EmbeddingMessage>(100);

        thread::spawn(move || {
            info!("Loading local embedding model on dedicated worker thread...");

            let model_res =
                SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                    .create_model();

            let model = match model_res {
                Ok(m) => m,
                Err(e) => {
                    error!("Failed to load local embedding model: {}", e);
                    let _ =
                        init_tx.send(Err(AppError::Embedding(format!("model load error: {e}"))));
                    return;
                }
            };

            info!("Local embedding model loaded and ready to process requests.");
            let _ = init_tx.send(Ok(()));

            while let Some((prompt, respond_to)) = rx.blocking_recv() {
                let result = model
                    .encode(&[prompt])
                    .map_err(|e| AppError::Embedding(format!("encode error: {e}")));

                let final_result = match result {
                    Ok(mut embeddings) => {
                        if embeddings.is_empty() {
                            Err(AppError::Embedding(
                                "model returned empty embedding array".into(),
                            ))
                        } else {
                            Ok(embeddings.remove(0))
                        }
                    }
                    Err(e) => Err(e),
                };

                let _ = respond_to.send(final_result);
            }
        });

        init_rx.recv().map_err(|_| {
            AppError::Embedding("Worker thread panicked during initialization".into())
        })??;

        Ok(Self { sender: tx })
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for LocalEmbeddingService {
    async fn encode(&self, prompt: String) -> Result<Vec<f32>, AppError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send((prompt, tx))
            .await
            .map_err(|_| AppError::Embedding("local embedding channel closed".into()))?;

        rx.await
            .map_err(|_| AppError::Embedding("local embedding dropped response".into()))?
    }
}

/// Ollama embedding provider using the external application.
#[derive(Clone)]
pub struct OllamaEmbeddingService {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

#[derive(Deserialize)]
struct OllamaEmbeddingResponse {
    embedding: Vec<f32>,
}

impl OllamaEmbeddingService {
    pub fn new(config: &AppConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client for embeddings");

        Self {
            client,
            base_url: config.ollama_base_url.clone(),
            model: config.ollama_embedding_model.clone(),
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OllamaEmbeddingService {
    async fn encode(&self, prompt: String) -> Result<Vec<f32>, AppError> {
        let url = format!("{}/api/embeddings", self.base_url);

        let json_body = serde_json::json!({
            "model": &self.model,
            "prompt": prompt,
        });

        let response = self
            .client
            .post(&url)
            .json(&json_body)
            .send()
            .await
            .map_err(|e| AppError::Embedding(format!("HTTP error: {e}")))?;

        match response.error_for_status() {
            Ok(resp) => {
                let body: OllamaEmbeddingResponse = resp
                    .json()
                    .await
                    .map_err(|e| AppError::Embedding(format!("JSON parse error: {e}")))?;

                if body.embedding.is_empty() {
                    Err(AppError::Embedding("Ollama returned empty embedding".into()))
                } else {
                    Ok(body.embedding)
                }
            }
            Err(e) => Err(AppError::Embedding(format!("Ollama API error: {e}"))),
        }
    }
}
