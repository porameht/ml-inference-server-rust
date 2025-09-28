use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub tokenizer_repo: String,
    pub revision: Option<String>,
    pub max_sequence_length: usize,
    pub device: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            tokenizer_repo: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            revision: None,
            max_sequence_length: 512,
            device: "cpu".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub text: String,
    pub normalize: bool,
}

impl EmbeddingRequest {
    pub fn new(text: String) -> Self {
        Self { text, normalize: true }
    }
    
    pub fn with_normalize(text: String, normalize: bool) -> Self {
        Self { text, normalize }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embedding: Vec<f32>,
    pub text: String,
    pub model_id: String,
}

#[derive(Debug, Clone)]
pub struct BatchEmbeddingRequest {
    pub texts: Vec<String>,
    pub normalize: bool,
}

impl BatchEmbeddingRequest {
    pub fn new(texts: Vec<String>) -> Self {
        Self { texts, normalize: true }
    }
    
    pub fn with_normalize(texts: Vec<String>, normalize: bool) -> Self {
        Self { texts, normalize }
    }
}

impl From<Vec<EmbeddingRequest>> for BatchEmbeddingRequest {
    fn from(requests: Vec<EmbeddingRequest>) -> Self {
        let texts = requests.iter().map(|r| r.text.clone()).collect();
        let normalize = requests.first().map(|r| r.normalize).unwrap_or(true);
        Self { texts, normalize }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub texts: Vec<String>,
    pub model_id: String,
}

impl BatchEmbeddingResponse {
    pub fn from_single_responses(responses: Vec<EmbeddingResponse>) -> Self {
        let embeddings = responses.iter().map(|r| r.embedding.clone()).collect();
        let texts = responses.iter().map(|r| r.text.clone()).collect();
        let model_id = responses.first().map(|r| r.model_id.clone()).unwrap_or_default();
        Self { embeddings, texts, model_id }
    }
}