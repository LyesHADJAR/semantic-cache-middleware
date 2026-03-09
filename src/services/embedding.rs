use crate::error::AppError;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use std::thread;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

/// Message sent to the embedding worker thread.
/// Tuple of (prompt, oneshot_sender_for_result)
pub type EmbeddingMessage = (String, oneshot::Sender<Result<Vec<f32>, AppError>>);

/// Handle to communicate with the background Embedding thread.
#[derive(Clone)]
pub struct EmbeddingService {
    sender: mpsc::Sender<EmbeddingMessage>,
}

impl EmbeddingService {
    pub fn init() -> Result<Self, AppError> {
        let (init_tx, init_rx) = std::sync::mpsc::sync_channel(1);
        let (tx, mut rx) = mpsc::channel::<EmbeddingMessage>(100);

        thread::spawn(move || {
            info!("Loading embedding model on dedicated worker thread...");

            let model_res =
                SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                    .create_model();

            let model = match model_res {
                Ok(m) => m,
                Err(e) => {
                    error!("Failed to load embedding model: {}", e);
                    let _ =
                        init_tx.send(Err(AppError::Embedding(format!("model load error: {e}"))));
                    return;
                }
            };

            info!("Embedding model loaded and ready to process requests.");
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

        // Wait for worker initialization to finish
        init_rx.recv().map_err(|_| {
            AppError::Embedding("Worker thread panicked during initialization".into())
        })??;

        Ok(Self { sender: tx })
    }

    /// Compute embedding for a single prompt asynchronously via the worker thread.
    pub async fn encode(&self, prompt: String) -> Result<Vec<f32>, AppError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send((prompt, tx))
            .await
            .map_err(|_| AppError::Embedding("embedding worker channel closed".into()))?;

        rx.await
            .map_err(|_| AppError::Embedding("embedding worker dropped response".into()))?
    }
}
