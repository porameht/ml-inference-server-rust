use std::sync::Arc;
use anyhow::Result;

use crate::domain::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse,
    EmbeddingService, ModelConfig,
};

pub struct EmbeddingUseCase {
    embedding_service: Arc<dyn EmbeddingService>,
}

impl EmbeddingUseCase {
    pub fn new(embedding_service: Arc<dyn EmbeddingService>) -> Self {
        Self { embedding_service }
    }

    pub async fn encode_single(&self, text: String, normalize: bool) -> Result<EmbeddingResponse> {
        let request = EmbeddingRequest::with_normalize(text, normalize);
        self.embedding_service.encode(request).await
    }

    pub async fn encode_batch(&self, texts: Vec<String>, normalize: bool) -> Result<BatchEmbeddingResponse> {
        let request = BatchEmbeddingRequest::with_normalize(texts, normalize);
        self.embedding_service.encode_batch(request).await
    }

    pub async fn get_model_info(&self) -> Result<ModelConfig> {
        self.embedding_service.get_model_info().await
    }
}

pub struct ModelManagementUseCase {
    embedding_service: Arc<dyn EmbeddingService>,
}

impl ModelManagementUseCase {
    pub fn new(embedding_service: Arc<dyn EmbeddingService>) -> Self {
        Self { embedding_service }
    }

    pub async fn switch_model(&self, config: ModelConfig) -> Result<()> {
        tracing::info!("Switching to model: {}", config.model_id);
        self.embedding_service.switch_model(config).await?;
        tracing::info!("Model switch completed successfully");
        Ok(())
    }

    pub async fn get_current_model(&self) -> Result<ModelConfig> {
        self.embedding_service.get_model_info().await
    }

    pub fn validate_model_config(&self, config: &ModelConfig) -> Result<()> {
        if config.model_id.is_empty() {
            return Err(anyhow::anyhow!("Model ID cannot be empty"));
        }
        
        if config.tokenizer_repo.is_empty() {
            return Err(anyhow::anyhow!("Tokenizer repository cannot be empty"));
        }

        if config.max_sequence_length == 0 || config.max_sequence_length > 8192 {
            return Err(anyhow::anyhow!(
                "Max sequence length must be between 1 and 8192, got: {}",
                config.max_sequence_length
            ));
        }

        Ok(())
    }
}