use anyhow::Result;
use std::sync::Arc;

use crate::application::{EmbeddingUseCase, ModelManagementUseCase};
use crate::domain::{EmbeddingService, ConfigurationService, ModelRepository};
use crate::infrastructure::{SentenceTransformerService, CandleModelLoader, FileConfigurationService};

/// Application services with dependency injection
pub struct ApplicationServices {
    pub embedding_use_case: EmbeddingUseCase,
    pub model_management_use_case: ModelManagementUseCase,
    // Keep references to services for access
    pub config_service: Arc<dyn ConfigurationService>,
    pub model_repository: Arc<dyn ModelRepository>,
    pub embedding_service: Arc<dyn EmbeddingService>,
}

impl ApplicationServices {
    /// Create application services with real Hugging Face model
    pub async fn new() -> Result<Self> {
        Self::create_with_real_services().await
    }

    /// Create with real implementations (production)
    pub async fn create_with_real_services() -> Result<Self> {
        tracing::info!("Creating application services with real Hugging Face model...");
        
        // Create dependencies
        let config_service: Arc<dyn ConfigurationService> = 
            Arc::new(FileConfigurationService::new()?);
        
        let model_loader = Arc::new(CandleModelLoader::new());
        let model_repository: Arc<dyn ModelRepository> = model_loader.clone();
        
        // Load model
        let config = config_service.get_model_config()?;
        model_repository.load_model(&config).await?;
        
        let embedding_service: Arc<dyn EmbeddingService> = 
            Arc::new(SentenceTransformerService::new(model_loader));
        
        // Create use cases
        let embedding_use_case = EmbeddingUseCase::new(embedding_service.clone());
        let model_management_use_case = ModelManagementUseCase::new(embedding_service.clone());
        
        tracing::info!("✅ Application services created with model: {}", config.model_id);
        
        Ok(Self {
            embedding_use_case,
            model_management_use_case,
            config_service,
            model_repository,
            embedding_service,
        })
    }

    /// Create with custom dependencies (for testing)
    pub fn with_dependencies(
        config_service: Arc<dyn ConfigurationService>,
        model_repository: Arc<dyn ModelRepository>,
        embedding_service: Arc<dyn EmbeddingService>,
    ) -> Result<Self> {
        let embedding_use_case = EmbeddingUseCase::new(embedding_service.clone());
        let model_management_use_case = ModelManagementUseCase::new(embedding_service.clone());
        
        Ok(Self {
            embedding_use_case,
            model_management_use_case,
            config_service,
            model_repository,
            embedding_service,
        })
    }
}