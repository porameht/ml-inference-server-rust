use std::sync::Arc;
use anyhow::Result;

use crate::domain::entities::{BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse};
use crate::domain::traits::{EmbeddingService, ModelRepository};

pub struct EmbeddingUseCase {
    embedding_service: Arc<dyn EmbeddingService>,
    model_repository: Arc<dyn ModelRepository>,
}

impl EmbeddingUseCase {
    pub fn new(
        embedding_service: Arc<dyn EmbeddingService>,
        model_repository: Arc<dyn ModelRepository>,
    ) -> Self {
        Self { 
            embedding_service,
            model_repository,
        }
    }

    /// Encode single text with business logic and validation
    pub async fn encode_single(&self, text: String, normalize: bool) -> Result<EmbeddingResponse> {
        // Business logic: validate input
        if text.trim().is_empty() {
            return Err(anyhow::anyhow!("Text cannot be empty"));
        }

        // Business logic: check if model is loaded
        let current_config = self.model_repository.get_current_config().await?;
        tracing::debug!("Using model: {} for encoding", current_config.model_id);

        let request = EmbeddingRequest::with_normalize(text, normalize);
        
        // Orchestrate: use embedding service for actual encoding
        let response = self.embedding_service.encode(request).await?;
        
        // Business logic: validate response
        if response.embedding.is_empty() {
            return Err(anyhow::anyhow!("Failed to generate embedding"));
        }

        tracing::debug!("Generated embedding with {} dimensions", response.embedding.len());
        Ok(response)
    }

    /// Encode batch with business logic and orchestration
    pub async fn encode_batch(&self, texts: Vec<String>, normalize: bool) -> Result<BatchEmbeddingResponse> {
        // Business logic: validate input
        if texts.is_empty() {
            return Err(anyhow::anyhow!("Text list cannot be empty"));
        }

        let non_empty_texts: Vec<String> = texts.into_iter()
            .filter(|text| !text.trim().is_empty())
            .collect();

        if non_empty_texts.is_empty() {
            return Err(anyhow::anyhow!("All texts are empty"));
        }

        // Business logic: check batch size limits
        const MAX_BATCH_SIZE: usize = 100;
        if non_empty_texts.len() > MAX_BATCH_SIZE {
            return Err(anyhow::anyhow!("Batch size {} exceeds maximum {}", non_empty_texts.len(), MAX_BATCH_SIZE));
        }

        // Business logic: ensure model is ready
        let current_config = self.model_repository.get_current_config().await?;
        tracing::debug!("Processing batch of {} texts with model: {}", non_empty_texts.len(), current_config.model_id);

        let request = BatchEmbeddingRequest::with_normalize(non_empty_texts, normalize);
        
        // Orchestrate: use embedding service for actual encoding
        let response = self.embedding_service.encode_batch(request).await?;
        
        // Business logic: validate response
        if response.embeddings.is_empty() {
            return Err(anyhow::anyhow!("Failed to generate any embeddings"));
        }

        tracing::debug!("Generated {} embeddings", response.embeddings.len());
        Ok(response)
    }

}