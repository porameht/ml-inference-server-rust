pub mod entities;
pub mod traits;
pub mod errors;

pub use entities::*;
pub use traits::*;
pub use errors::*;

/// Dependency injection container interface
pub trait ServiceContainer: Send + Sync {
    type EmbeddingService: EmbeddingService;
    type ModelRepository: ModelRepository;
    type ConfigurationService: ConfigurationService;

    fn embedding_service(&self) -> &Self::EmbeddingService;
    fn model_repository(&self) -> &Self::ModelRepository;
    fn configuration_service(&self) -> &Self::ConfigurationService;
}