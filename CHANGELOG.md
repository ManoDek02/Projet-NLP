# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Advanced RAG components (reranking, caching, conversation memory)
- Rate limiting middleware
- Prometheus metrics endpoint
- OpenTelemetry tracing support

### Changed
- Improved error handling in LLM service

### Fixed
- Memory leak in batch embedding processing

---

## [1.0.0] - 2024-01-15

### Added
- Initial release of Reddit RAG Chatbot
- FastAPI REST API with versioned endpoints
- Gradio web interface
- Streamlit alternative interface
- Sentence-transformers multilingual embeddings
- ChromaDB vector store integration
- Multi-provider LLM support (Ollama, OpenAI, Anthropic)
- Docker deployment configuration
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions
- Documentation (README, API docs, architecture)

### Core Features
- **Embedding Service**: `paraphrase-multilingual-MiniLM-L12-v2` model
- **Vector Store**: ChromaDB with persistent storage
- **LLM Integration**: Ollama (default), OpenAI, Anthropic
- **Data Processing**: CSV to JSON conversion, text cleaning
- **Indexing**: Batch embedding and vector indexing

### API Endpoints
- `POST /api/v1/chat/` - Chat with the bot
- `GET /api/v1/chat/stats` - Get statistics
- `GET /api/v1/chat/examples` - Get example questions
- `GET /api/v1/health/` - Health check
- `GET /api/v1/health/ready` - Readiness probe
- `GET /api/v1/health/live` - Liveness probe

### Data
- 56,295 Reddit conversations indexed
- Multilingual support (French/English focus)
- 384-dimensional embeddings

---

## [0.2.0] - 2024-01-01

### Added
- Streamlit interface as alternative UI
- Benchmark script for performance testing
- Input validation and sanitization
- Logging configuration with Loguru

### Changed
- Refactored embedding service for better batch processing
- Improved error messages

### Fixed
- Unicode handling in text processor
- Memory usage during large batch indexing

---

## [0.1.0] - 2023-12-15

### Added
- Basic RAG pipeline implementation
- Simple FastAPI endpoint
- ChromaDB vector store setup
- Initial Gradio interface
- Data preparation script

### Known Issues
- No authentication
- No rate limiting
- Limited error handling

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2024-01-15 | Production-ready release |
| 0.2.0 | 2024-01-01 | Added Streamlit, benchmarking |
| 0.1.0 | 2023-12-15 | Initial development release |

---

## Upgrade Notes

### Upgrading to 1.0.0

1. Update dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Re-index conversations (schema changes):
   ```bash
   python scripts/index_conversations.py --reset
   ```

3. Update environment variables:
   ```bash
   cp .env.example .env
   # Update with new variables
   ```

### Breaking Changes in 1.0.0

- API endpoint prefix changed from `/chat` to `/api/v1/chat`
- Response schema updated with `metadata` field
- Configuration moved from `config.yaml` to environment variables
