use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model not found: {model_id}")]
    ModelNotFound { model_id: String },
    
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },
    
    #[error("Encoding failed: {message}")]
    EncodingFailed { message: String },
    
    #[error("Model loading failed: {message}")]
    ModelLoadFailed { message: String },
    
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, InferenceError>;