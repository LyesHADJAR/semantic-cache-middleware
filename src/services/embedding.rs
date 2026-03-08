use crate::error::AppError;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use std::thread;
use tokio::sync::{mpsc, oneshot};

/// Message sent to the embedding worker thread.
/// Tuple of (prompt, oneshot_sender_for_result)
pub type EmbeddingMessage = (String, oneshot::Sender<Result<Vec<f32>, AppError>>);

/// Handle to communicate with the background Embedding thread.
#[derive(Clone)]
pub struct EmbeddingService {
    sender: mpsc::Sender<EmbeddingMessage>,
}

impl EmbeddingService {
    pub fn new() -> Self {
        // Create an mpsc channel for communication
        let (tx, mut rx) = mpsc::channel::<EmbeddingMessage>(100);

        // Spawn a dedicated native OS thread for CPU-heavy work
        // this prevents blocking the Tokio async executor.
        thread::spawn(move || {
            println!("Loading embedding model on dedicated worker thread...");

            // Note: rust-bert's create_model can fail, but since we are in background,
            // we should unwrap or handle gracefully. Let's unwrap as failure to load is fatal.
            let model =
                SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                    .create_model()
                    .expect("Failed to load embedding model");

            println!("Embedding model loaded and ready to process requests.");

            // Blocking receive loop
            while let Some((prompt, respond_to)) = rx.blocking_recv() {
                // Compute embedding (batch size of 1)
                let result = model
                    .encode(&[prompt])
                    .map_err(|e| AppError::Embedding(format!("encode error: {}", e)));

                // rust-bert encode returns Vec<Vec<f32>>, we want the first element
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

                // Send back the result (ignore error if receiver dropped it)
                let _ = respond_to.send(final_result);
            }
        });

        Self { sender: tx }
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
