use std::sync::Arc;
use anyhow::{anyhow, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;
use tokio::sync::RwLock;

use crate::domain::entities::ModelConfig;
use crate::domain::traits::ModelRepository;

pub struct ModelComponents {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub config: ModelConfig,
}

pub struct CandleModelLoader {
    current_model: Arc<RwLock<Option<ModelComponents>>>,
}

impl CandleModelLoader {
    pub fn new() -> Self {
        Self {
            current_model: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn get_model(&self) -> Result<Arc<RwLock<Option<ModelComponents>>>> {
        Ok(self.current_model.clone())
    }

    async fn download_and_load_model(&self, config: &ModelConfig) -> Result<ModelComponents> {
        tracing::info!("Loading model: {}", config.model_id);
        tracing::debug!("Model config: {:?}", config);

        let device = self.get_device(&config.device)?;
        
        let api = Api::new()
            .map_err(|e| anyhow!("Failed to create HuggingFace API client: {}", e))?;
        
        tracing::debug!("Created HF API client successfully");
        
        let repo = api.model(config.model_id.clone());
        tracing::debug!("Created model repository handle for: {}", config.model_id);

        let tokenizer_filename = repo.get("tokenizer.json").await?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let config_filename = repo.get("config.json").await?;
        let config_content = std::fs::read_to_string(config_filename)?;
        let bert_config: BertConfig = serde_json::from_str(&config_content)?;

        let weights_filename = match repo.get("pytorch_model.bin").await {
            Ok(path) => path,
            Err(_) => repo.get("model.safetensors").await?,
        };

        let weights = candle_core::safetensors::load(&weights_filename, &device)?;
        let var_builder = VarBuilder::from_tensors(weights, candle_core::DType::F32, &device);

        let model = BertModel::load(var_builder, &bert_config)?;

        Ok(ModelComponents {
            model,
            tokenizer,
            device,
            config: config.clone(),
        })
    }

    fn get_device(&self, device_str: &str) -> Result<Device> {
        match device_str.to_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" | "gpu" => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Device::new_cuda(0)?)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    tracing::warn!("CUDA requested but not available, falling back to CPU");
                    Ok(Device::Cpu)
                }
            }
            "metal" => {
                #[cfg(feature = "metal")]
                {
                    Ok(Device::new_metal(0)?)
                }
                #[cfg(not(feature = "metal"))]
                {
                    tracing::warn!("Metal requested but not available, falling back to CPU");
                    Ok(Device::Cpu)
                }
            }
            _ => {
                tracing::warn!("Unknown device '{}', falling back to CPU", device_str);
                Ok(Device::Cpu)
            }
        }
    }
}

#[async_trait::async_trait]
impl ModelRepository for CandleModelLoader {
    async fn load_model(&self, config: &ModelConfig) -> Result<()> {
        let components = self.download_and_load_model(config).await?;
        let mut model_guard = self.current_model.write().await;
        *model_guard = Some(components);
        tracing::info!("Model loaded successfully: {}", config.model_id);
        Ok(())
    }

    async fn get_current_config(&self) -> Result<ModelConfig> {
        let model_guard = self.current_model.read().await;
        match model_guard.as_ref() {
            Some(components) => Ok(components.config.clone()),
            None => Err(anyhow!("No model loaded")),
        }
    }
}