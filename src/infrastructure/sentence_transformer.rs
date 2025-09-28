#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

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

        if texts.len() == 1 {
            // Single text encoding
            self.encode_single_text(&texts[0], components, normalize).await
        } else {
            // Batch encoding for better performance
            self.encode_batch_texts(texts, components, normalize).await
        }
    }

    async fn encode_single_text(&self, text: &str, components: &crate::infrastructure::model_loader::ModelComponents, normalize: bool) -> Result<Vec<Vec<f32>>> {
        let encoding = components.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let tokens = encoding.get_ids().to_vec();
        let token_ids = Tensor::new(&tokens[..], &components.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let ys = components.model.forward(&token_ids, &token_type_ids, None)?;
        
        let embedding = if normalize {
            self.normalize_l2(&ys)?
        } else {
            ys
        };

        let embedding_vec = embedding.to_vec1::<f32>()?;
        Ok(vec![embedding_vec])
    }

    async fn encode_batch_texts(&self, texts: &[String], components: &crate::infrastructure::model_loader::ModelComponents, normalize: bool) -> Result<Vec<Vec<f32>>> {
        let tokens = components.tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("Batch tokenization failed: {}", e))?;

        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &components.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &components.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        tracing::debug!("Running inference on batch {:?}", token_ids.shape());
        let embeddings = components.model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        tracing::debug!("Generated embeddings {:?}", embeddings.shape());

        // Apply mean pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let pooled_embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        
        let final_embeddings = if normalize {
            self.normalize_l2(&pooled_embeddings)?
        } else {
            pooled_embeddings
        };

        tracing::debug!("Pooled embeddings {:?}", final_embeddings.shape());

        // Convert to Vec<Vec<f32>>
        let mut result = Vec::new();
        for i in 0..texts.len() {
            let embedding = final_embeddings.get(i)?;
            let embedding_vec = embedding.to_vec1::<f32>()?;
            result.push(embedding_vec);
        }

        Ok(result)
    }

    fn normalize_l2(&self, v: &Tensor) -> Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
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