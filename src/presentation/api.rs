use std::sync::Arc;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::application::ApplicationServices;

#[derive(Debug, Deserialize)]
pub struct EncodeRequest {
    pub text: String,
    #[serde(default = "default_normalize")]
    pub normalize: bool,
}

#[derive(Debug, Deserialize)]
pub struct BatchEncodeRequest {
    pub texts: Vec<String>,
    #[serde(default = "default_normalize")]
    pub normalize: bool,
}

#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
        }
    }
}

type ApiResult<T> = Result<Json<ApiResponse<T>>, StatusCode>;

fn handle_result<T>(result: anyhow::Result<T>) -> ApiResult<T> {
    match result {
        Ok(data) => Ok(Json(ApiResponse::success(data))),
        Err(e) => {
            tracing::error!("API error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

fn default_normalize() -> bool {
    true
}

pub fn create_router(services: Arc<ApplicationServices>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/encode", post(encode_single))
        .route("/encode/batch", post(encode_batch))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(services)
}

async fn health_check() -> Json<ApiResponse<&'static str>> {
    Json(ApiResponse::success("Sentence Transformer API is running"))
}

async fn encode_single(
    State(services): State<Arc<ApplicationServices>>,
    Json(request): Json<EncodeRequest>,
) -> ApiResult<crate::domain::EmbeddingResponse> {
    let result = services.embedding_use_case
        .encode_single(request.text, request.normalize)
        .await;
    handle_result(result)
}

async fn encode_batch(
    State(services): State<Arc<ApplicationServices>>,
    Json(request): Json<BatchEncodeRequest>,
) -> ApiResult<crate::domain::BatchEmbeddingResponse> {
    let result = services.embedding_use_case
        .encode_batch(request.texts, request.normalize)
        .await;
    handle_result(result)
}

