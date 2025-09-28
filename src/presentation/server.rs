use std::sync::Arc;
use anyhow::Result;
use axum::Router;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;

use crate::application::ApplicationServices;
use crate::infrastructure::ServerConfig;
use crate::presentation::create_router;

pub struct InferenceServer {
    app: Router,
    config: ServerConfig,
}

impl InferenceServer {
    pub fn new(services: Arc<ApplicationServices>, config: ServerConfig) -> Self {
        let app = create_router(services)
            .layer(TraceLayer::new_for_http());

        Self { app, config }
    }

    pub async fn start(self) -> Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        
        tracing::info!("🚀 Starting Sentence Transformer API server");
        tracing::info!("   📍 Address: http://{}", addr);
        tracing::info!("   🎯 Endpoints:");
        tracing::info!("      GET  /health           - Health check");
        tracing::info!("      POST /encode           - Single text encoding");
        tracing::info!("      POST /encode/batch     - Batch text encoding");
        tracing::info!("      GET  /model/info       - Current model information");
        tracing::info!("      POST /model/switch     - Switch to different model");

        let listener = TcpListener::bind(&addr).await?;
        
        tracing::info!("✅ Server listening on http://{}", addr);
        
        axum::serve(listener, self.app).await?;
        
        Ok(())
    }
}