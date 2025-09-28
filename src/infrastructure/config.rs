use anyhow::Result;
use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use crate::domain::{ConfigurationService, ModelConfig};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    pub model: ModelConfig,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            workers: 4,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            server: ServerConfig::default(),
        }
    }
}

pub struct FileConfigurationService {
    config: Arc<RwLock<AppConfig>>,
}

impl FileConfigurationService {
    pub fn new() -> Result<Self, ConfigError> {
        Self::new_with_environment(None)
    }

    pub fn new_with_environment(env: Option<&str>) -> Result<Self, ConfigError> {
        let default_env = std::env::var("INFERENCE_ENV").unwrap_or_else(|_| "development".to_string());
        let environment = env.unwrap_or(&default_env);
        
        let mut builder = Config::builder()
            .add_source(File::with_name("config/default").required(false));

        // Add environment-specific config file
        if environment != "default" {
            builder = builder.add_source(
                File::with_name(&format!("config/{}", environment)).required(false)
            );
        }

        // Add local override file
        builder = builder.add_source(File::with_name("config/local").required(false));

        // Add environment variables
        builder = builder.add_source(Environment::with_prefix("INFERENCE"));

        let settings = builder.build()?;

        let config: AppConfig = match settings.try_deserialize() {
            Ok(config) => {
                tracing::info!("Loaded configuration for environment: {}", environment);
                config
            }
            Err(e) => {
                tracing::warn!("Failed to deserialize config for environment '{}', using defaults: {}", environment, e);
                AppConfig::default()
            }
        };

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
        })
    }

    pub fn get_app_config(&self) -> Result<AppConfig> {
        let config = self.config.read().map_err(|_| {
            anyhow::anyhow!("Failed to acquire read lock on configuration")
        })?;
        Ok(config.clone())
    }

    pub fn get_server_config(&self) -> Result<ServerConfig> {
        let config = self.config.read().map_err(|_| {
            anyhow::anyhow!("Failed to acquire read lock on configuration")
        })?;
        Ok(config.server.clone())
    }
}

impl ConfigurationService for FileConfigurationService {
    fn get_model_config(&self) -> Result<ModelConfig> {
        let config = self.config.read().map_err(|_| {
            anyhow::anyhow!("Failed to acquire read lock on configuration")
        })?;
        Ok(config.model.clone())
    }

    fn update_model_config(&self, model_config: ModelConfig) -> Result<()> {
        let mut config = self.config.write().map_err(|_| {
            anyhow::anyhow!("Failed to acquire write lock on configuration")
        })?;
        config.model = model_config;
        Ok(())
    }
}