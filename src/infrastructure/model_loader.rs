#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::sync::Arc;
use anyhow::{anyhow, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{Tokenizer, PaddingParams};
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
        
        let (default_model, default_revision) = self.get_default_model_config();
        let (model_id, revision) = if config.model_id.is_empty() {
            (default_model, default_revision)
        } else {
            (config.model_id.clone(), config.revision.clone().unwrap_or_else(|| "main".to_string()))
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config_file = api.get("config.json")?;
            let tokenizer_file = api.get("tokenizer.json")?;
            let weights = if config.use_pth.unwrap_or(false) {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config_file, tokenizer_file, weights)
        };

        let config_content = std::fs::read_to_string(config_filename)?;
        let mut bert_config: BertConfig = serde_json::from_str(&config_content)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Configure tokenizer for batch processing
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let vb = if config.use_pth.unwrap_or(false) {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };

        if config.approximate_gelu.unwrap_or(false) {
            bert_config.hidden_act = HiddenAct::GeluApproximate;
        }

        let model = BertModel::load(vb, &bert_config)?;

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

    fn get_default_model_config(&self) -> (String, String) {
        ("sentence-transformers/all-MiniLM-L6-v2".to_string(), "refs/pr/21".to_string())
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