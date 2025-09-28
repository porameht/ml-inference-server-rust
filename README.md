# Sentence Transformer Inference Service

A flexible and efficient Rust-based inference service for sentence transformer models using the Candle ML framework. Built with clean architecture principles and designed for concurrent access with ArcRwLock.

## Features

- 🚀 **Fast Inference**: Powered by Candle ML framework for efficient CPU/GPU inference
- 🔄 **Model Switching**: Runtime model switching without service restart
- 🏗️ **Clean Architecture**: Domain-driven design with clear separation of concerns
- 🔒 **Thread Safe**: Concurrent model access using `Arc<RwLock<T>>`
- 📝 **Multiple Formats**: Support for JSON, CSV, and human-readable output
- 🌐 **Multiple Models**: Pre-configured popular sentence transformer models
- ⚙️ **Configurable**: TOML-based configuration with environment variable support

## Quick Start

### Prerequisites

- Rust 1.70+ (2021 edition)
- Optional: CUDA toolkit for GPU acceleration

### Installation

#### Option 1: Local Development

```bash
# Clone and build
git clone <your-repo>
cd inference
cargo build --release

# Start the API server
cargo run
```

#### Option 2: Docker

```bash
# Using docker-compose (recommended)
docker-compose up --build

# Or using Docker directly
docker build -t inference-api .
docker run -p 8080:8080 inference-api
```

The server will start on `http://127.0.0.1:8080` by default.

## API Usage

### Health Check

```bash
curl http://localhost:8080/health
```

### Single Text Encoding

```bash
curl -X POST http://localhost:8080/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "normalize": true}'
```

### Batch Text Encoding

```bash
curl -X POST http://localhost:8080/encode/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Text 1", "Text 2", "Text 3"],
    "normalize": true
  }'
```

### Model Management

```bash
# Get current model info
curl http://localhost:8080/model/info

# Switch model
curl -X POST http://localhost:8080/model/switch \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "sentence-transformers/all-mpnet-base-v2",
    "tokenizer_repo": "sentence-transformers/all-mpnet-base-v2",
    "max_sequence_length": 512,
    "device": "cpu"
  }'
```

See [examples/api_usage.md](examples/api_usage.md) for detailed API documentation and client examples.

## Architecture

The project follows clean architecture principles:

```
src/
├── domain/           # Business logic and entities
│   ├── entities.rs   # Core data structures
│   └── traits.rs     # Domain interfaces
├── infrastructure/   # External dependencies
│   ├── config.rs     # Configuration management
│   ├── model_loader.rs  # Candle model loading
│   └── sentence_transformer.rs  # ML inference
├── application/      # Use cases and orchestration
│   ├── services.rs   # Service composition
│   └── use_cases.rs  # Business use cases
└── presentation/     # User interfaces
    ├── cli.rs        # Command-line interface
    └── handlers.rs   # Command handlers
```

## Configuration

### Default Configuration

Edit `config/default.toml`:

```toml
[model]
model_id = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer_repo = "sentence-transformers/all-MiniLM-L6-v2"
max_sequence_length = 512
device = "cpu"

[server]
host = "127.0.0.1"
port = 8080
workers = 4
```

### Environment Variables

Override configuration with environment variables:

```bash
export INFERENCE_MODEL__MODEL_ID="sentence-transformers/all-mpnet-base-v2"
export INFERENCE_MODEL__DEVICE="cuda"
export INFERENCE_SERVER__PORT="3000"
```

### Pre-configured Models

See `config/models.toml` for ready-to-use model configurations:

- `all-MiniLM-L6-v2`: Fast, general-purpose (default)
- `all-mpnet-base-v2`: Higher quality, slower
- `multilingual-e5-base`: 100+ languages support
- `bge-small-en-v1.5`: Compact English model
- `gte-large`: Superior performance, larger size

## Examples

### Semantic Search

```bash
# Encode query and documents
cargo run -- encode -t "machine learning algorithms" --output-format json > query.json
cargo run -- encode-batch \
  -t "supervised learning with neural networks" \
  -t "unsupervised clustering techniques" \
  -t "reinforcement learning applications" \
  --output-format json > docs.json
```

### Model Comparison

```bash
# Compare embeddings from different models
cargo run -- encode -t "artificial intelligence" --output-format json
cargo run -- switch-model -m "sentence-transformers/all-mpnet-base-v2"
cargo run -- encode -t "artificial intelligence" --output-format json
```

## Performance

### GPU Acceleration

Enable CUDA support:

```bash
# Build with CUDA
cargo build --release --features cuda

# Use GPU
cargo run -- switch-model -m "your-model" --device cuda
```

### Concurrent Access

The service uses `Arc<RwLock<T>>` for thread-safe model access:

- Multiple readers can access the model simultaneously
- Model switching requires exclusive write access
- Zero-copy model sharing between threads

## Development

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# With specific features
cargo build --features cuda
cargo build --features metal  # macOS
cargo build --features mkl    # Intel MKL
```

### Testing

```bash
# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo test
```

### Linting

```bash
cargo clippy
cargo fmt
```

## Dependencies

- **Candle**: ML framework for Rust
- **Tokenizers**: HuggingFace tokenizers
- **Tokio**: Async runtime
- **Clap**: Command-line parsing
- **Serde**: Serialization
- **Anyhow**: Error handling
- **Tracing**: Structured logging

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Roadmap

- [ ] HTTP REST API server
- [ ] WebSocket streaming
- [ ] Model quantization support
- [ ] Batch processing optimization
- [ ] Metrics and monitoring
- [ ] Docker containerization
- [ ] Kubernetes deployment