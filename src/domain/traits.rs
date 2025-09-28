use anyhow::Result;
use async_trait::async_trait;

use super::entities::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse,
    ModelConfig,
};

#[async_trait]
pub trait EmbeddingService: Send + Sync {
    async fn encode(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>;
    async fn encode_batch(&self, request: BatchEmbeddingRequest) -> Result<BatchEmbeddingResponse>;
    async fn get_model_info(&self) -> Result<ModelConfig>;
    async fn switch_model(&self, config: ModelConfig) -> Result<()>;
}

#[async_trait]
pub trait ModelRepository: Send + Sync {
    async fn load_model(&self, config: &ModelConfig) -> Result<()>;
    async fn get_current_config(&self) -> Result<ModelConfig>;
}

pub trait ConfigurationService: Send + Sync {
    fn get_model_config(&self) -> Result<ModelConfig>;
    fn update_model_config(&self, config: ModelConfig) -> Result<()>;
}