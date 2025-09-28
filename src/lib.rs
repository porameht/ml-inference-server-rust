pub mod domain;
pub mod infrastructure;
pub mod application;
pub mod presentation;

pub use domain::*;
pub use infrastructure::*;
pub use application::*;
pub use presentation::*;


/// Simple but flexible dependency injection container
pub struct Container {
    pub config_service: std::sync::Arc<dyn domain::ConfigurationService>,
    pub model_repository: std::sync::Arc<dyn domain::ModelRepository>,
    pub embedding_service: std::sync::Arc<dyn domain::EmbeddingService>,
}

impl Container {
    /// Create container with real implementations
    pub async fn new() -> anyhow::Result<Self> {
        Self::with_real_services().await
    }

    /// Create container with real Hugging Face model
    pub async fn with_real_services() -> anyhow::Result<Self> {
        tracing::info!("Creating container with real Hugging Face model...");
        
        let config_service: std::sync::Arc<dyn domain::ConfigurationService> = 
            std::sync::Arc::new(infrastructure::FileConfigurationService::new()?);
        
        let model_loader = std::sync::Arc::new(infrastructure::CandleModelLoader::new());
        let model_repository: std::sync::Arc<dyn domain::ModelRepository> = 
            model_loader.clone();
        
        // Load the model
        let config = config_service.get_model_config()?;
        model_repository.load_model(&config).await?;
        
        let embedding_service: std::sync::Arc<dyn domain::EmbeddingService> = 
            std::sync::Arc::new(infrastructure::SentenceTransformerService::new(model_loader));
        
        tracing::info!("✅ Model loaded: {}", config.model_id);
        
        Ok(Self {
            config_service,
            model_repository,
            embedding_service,
        })
    }

    /// Create container with custom dependencies (for testing)
    pub fn with_dependencies(
        config_service: std::sync::Arc<dyn domain::ConfigurationService>,
        model_repository: std::sync::Arc<dyn domain::ModelRepository>,
        embedding_service: std::sync::Arc<dyn domain::EmbeddingService>,
    ) -> Self {
        Self {
            config_service,
            model_repository,
            embedding_service,
        }
    }

    /// Create a builder for more flexible container construction
    pub fn builder() -> ContainerBuilder {
        ContainerBuilder::new()
    }
}

/// Builder for flexible dependency injection container construction
pub struct ContainerBuilder {
    config_service: Option<std::sync::Arc<dyn domain::ConfigurationService>>,
    model_repository: Option<std::sync::Arc<dyn domain::ModelRepository>>,
    embedding_service: Option<std::sync::Arc<dyn domain::EmbeddingService>>,
}

impl ContainerBuilder {
    pub fn new() -> Self {
        Self {
            config_service: None,
            model_repository: None,
            embedding_service: None,
        }
    }

    pub fn with_config_service(mut self, service: std::sync::Arc<dyn domain::ConfigurationService>) -> Self {
        self.config_service = Some(service);
        self
    }

    pub fn with_model_repository(mut self, repository: std::sync::Arc<dyn domain::ModelRepository>) -> Self {
        self.model_repository = Some(repository);
        self
    }

    pub fn with_embedding_service(mut self, service: std::sync::Arc<dyn domain::EmbeddingService>) -> Self {
        self.embedding_service = Some(service);
        self
    }

    pub async fn build(self) -> anyhow::Result<Container> {
        let config_service = if let Some(service) = self.config_service {
            service
        } else {
            std::sync::Arc::new(infrastructure::FileConfigurationService::new()?)
        };

        let model_repository = if let Some(repository) = self.model_repository {
            repository
        } else {
            std::sync::Arc::new(infrastructure::CandleModelLoader::new())
        };

        let embedding_service = if let Some(service) = self.embedding_service {
            service
        } else {
            // Load model and create service
            let config = config_service.get_model_config()?;
            model_repository.load_model(&config).await?;
            
            let model_loader = std::sync::Arc::new(infrastructure::CandleModelLoader::new());
            model_loader.load_model(&config).await?;
            
            std::sync::Arc::new(infrastructure::SentenceTransformerService::new(model_loader))
        };

        Ok(Container {
            config_service,
            model_repository,
            embedding_service,
        })
    }
}

/// Re-export commonly used types for easier access
pub mod prelude {
    pub use crate::domain::{EmbeddingService, ModelRepository, ConfigurationService, ModelConfig};
    pub use crate::application::ApplicationServices;
    pub use crate::{Container, ContainerBuilder};
    pub use std::sync::Arc;
}