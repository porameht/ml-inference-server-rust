pub mod domain;
pub mod infrastructure;
pub mod application;
pub mod presentation;

use crate::application::use_cases::EmbeddingUseCase;
use crate::domain::traits::{ConfigurationService, ModelRepository, EmbeddingService};
use crate::infrastructure::config::FileConfigurationService;
use crate::infrastructure::model_loader::CandleModelLoader;
use crate::infrastructure::sentence_transformer::SentenceTransformerService;

pub struct DiContainer {
    pub embedding_use_case: std::sync::Arc<EmbeddingUseCase>,
}

impl DiContainer {
    /// Create container with all dependencies wired up
    pub async fn new() -> anyhow::Result<Self> {
        let embedding_use_case = create_embedding_use_case().await?;
        
        Ok(Self {
            embedding_use_case,
        })
    }

    /// Create container with custom config path
    pub async fn with_config(config_path: Option<&str>) -> anyhow::Result<Self> {
        let embedding_use_case = create_embedding_use_case_with_config(config_path).await?;
        
        Ok(Self {
            embedding_use_case,
        })
    }
}

/// Factory function to create EmbeddingUseCase with all dependencies
pub async fn create_embedding_use_case() -> anyhow::Result<std::sync::Arc<EmbeddingUseCase>> {
    create_embedding_use_case_with_config(None).await
}

/// Factory function to create EmbeddingUseCase with custom config
pub async fn create_embedding_use_case_with_config(config_path: Option<&str>) -> anyhow::Result<std::sync::Arc<EmbeddingUseCase>> {
    tracing::info!("Creating dependency injection container...");
    
    // Create infrastructure dependencies
    let config_service: std::sync::Arc<dyn ConfigurationService> = 
        if let Some(_path) = config_path {
            // For now, use environment-based config instead of path-based
            std::sync::Arc::new(FileConfigurationService::new_with_environment(Some("custom"))?)
        } else {
            std::sync::Arc::new(FileConfigurationService::new()?)
        };
    
    let model_loader = std::sync::Arc::new(CandleModelLoader::new());
    let model_repository: std::sync::Arc<dyn ModelRepository> = 
        model_loader.clone();
    
    // Load initial model
    let config = config_service.get_model_config()?;
    model_repository.load_model(&config).await?;
    
    let embedding_service: std::sync::Arc<dyn EmbeddingService> = 
        std::sync::Arc::new(SentenceTransformerService::new(model_loader));
    
    // Wire up use case with dependencies (Clean Architecture DI)
    let embedding_use_case = std::sync::Arc::new(EmbeddingUseCase::new(
        embedding_service,
        model_repository,
    ));
    
    tracing::info!("✅ Dependency container ready with model: {}", config.model_id);
    
    Ok(embedding_use_case)
}