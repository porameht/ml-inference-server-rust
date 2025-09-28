mod domain;
mod infrastructure;
mod application;
mod presentation;

use std::sync::Arc;
use anyhow::Result;
use tracing_subscriber::EnvFilter;

use application::ApplicationServices;
use presentation::InferenceServer;

#[tokio::main]
async fn main() -> Result<()> {
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

    // Create services with real Hugging Face models
    let services = ApplicationServices::new().await?;

    let server_config = infrastructure::ServerConfig::default();
    let server = InferenceServer::new(Arc::new(services), server_config);
    
    server.start().await?;

    Ok(())
}
