use std::sync::Arc;
use anyhow::{anyhow, Result};
use candle_core::Tensor;

use crate::domain::entities::{
    BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest, EmbeddingResponse, ModelConfig,
};
use crate::domain::traits::{EmbeddingService, ModelRepository};
use crate::infrastructure::model_loader::CandleModelLoader;

pub struct SentenceTransformerService {
    model_loader: Arc<CandleModelLoader>,
}

impl SentenceTransformerService {
    pub fn new(model_loader: Arc<CandleModelLoader>) -> Self {
        Self { model_loader }
    }

    async fn encode_texts(&self, texts: &[String], normalize: bool) -> Result<Vec<Vec<f32>>> {
        let model_ref = self.model_loader.get_model().await?;
        let model_guard = model_ref.read().await;
        
        let components = model_guard
            .as_ref()
            .ok_or_else(|| anyhow!("No model loaded"))?;

        let mut embeddings = Vec::new();

        for text in texts {
            let encoding = components
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

            let tokens = encoding.get_ids();
            let token_ids = Tensor::new(tokens, &components.device)?
                .unsqueeze(0)?;

            let attention_mask = Tensor::ones(
                (1, tokens.len()),
                candle_core::DType::U8,
                &components.device,
            )?;

            let token_type_ids = Tensor::zeros(
                (1, tokens.len()),
                candle_core::DType::U8,
                &components.device,
            )?;

            let outputs = components.model.forward(
                &token_ids,
                &token_type_ids,
                Some(&attention_mask),
            )?;

            let pooled_output = self.mean_pooling(&outputs, &attention_mask)?;
            
            let embedding = if normalize {
                self.normalize_tensor(&pooled_output)?
            } else {
                pooled_output
            };

            let embedding_vec = embedding.to_vec1::<f32>()?;
            embeddings.push(embedding_vec);
        }

        Ok(embeddings)
    }

    fn mean_pooling(&self, token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let attention_mask = attention_mask.to_dtype(candle_core::DType::F32)?;
        let attention_mask = attention_mask.unsqueeze(2)?;
        let attention_mask = attention_mask.expand(token_embeddings.shape())?;

        let masked_embeddings = token_embeddings.mul(&attention_mask)?;
        let sum_embeddings = masked_embeddings.sum(1)?;
        let sum_mask = attention_mask.sum(1)?;
        
        Ok(sum_embeddings.div(&sum_mask)?)
    }

    fn normalize_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let norm = tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
        Ok(tensor.div(&norm)?)
    }
}

#[async_trait::async_trait]
impl EmbeddingService for SentenceTransformerService {
    async fn encode(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let embeddings = self.encode_texts(&[request.text.clone()], request.normalize).await?;
        let config = self.model_loader.get_current_config().await?;
        
        Ok(EmbeddingResponse {
            embedding: embeddings.into_iter().next().unwrap(),
            text: request.text,
            model_id: config.model_id,
        })
    }

    async fn encode_batch(&self, request: BatchEmbeddingRequest) -> Result<BatchEmbeddingResponse> {
        let embeddings = self.encode_texts(&request.texts, request.normalize).await?;
        let config = self.model_loader.get_current_config().await?;
        
        Ok(BatchEmbeddingResponse {
            embeddings,
            texts: request.texts,
            model_id: config.model_id,
        })
    }

    async fn get_model_info(&self) -> Result<ModelConfig> {
        self.model_loader.get_current_config().await
    }

    async fn switch_model(&self, config: ModelConfig) -> Result<()> {
        self.model_loader.load_model(&config).await
    }
}