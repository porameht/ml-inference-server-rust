use anyhow::Result;
use tracing_subscriber::EnvFilter;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;

use inference::{DiContainer, infrastructure::config::ServerConfig, presentation::api::create_router};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    tracing::info!("🤖 Initializing Sentence Transformer Inference Service");

    // Create DI container with all dependencies
    let container = DiContainer::new().await?;

    // Create router and server
    let server_config = ServerConfig::default();
    let app = create_router(container.embedding_use_case)
        .layer(TraceLayer::new_for_http());

    let addr = format!("{}:{}", server_config.host, server_config.port);
    
    tracing::info!("🚀 Starting Sentence Transformer API server");
    tracing::info!("   📍 Address: http://{}", addr);
    tracing::info!("   🎯 Endpoints:");
    tracing::info!("      GET  /health           - Health check");
    tracing::info!("      POST /encode           - Single text encoding");
    tracing::info!("      POST /encode/batch     - Batch text encoding");

    let listener = TcpListener::bind(&addr).await?;
    
    tracing::info!("✅ Server listening on http://{}", addr);
    
    axum::serve(listener, app).await?;

    Ok(())
}
