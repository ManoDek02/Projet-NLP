# Reddit RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot built on Reddit conversation data with multilingual support.

## Features

- **RAG Pipeline**: Semantic search with vector embeddings + LLM generation
- **Multilingual Support**: 60+ languages via `paraphrase-multilingual-MiniLM-L12-v2`
- **Multiple LLM Providers**: Ollama (local), OpenAI, Anthropic
- **Production Ready**: Docker, health checks, logging, monitoring
- **Multiple Interfaces**: REST API, Gradio UI, Streamlit UI
- **56,000+ Reddit Conversations**: Pre-indexed dataset

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (optional, for local LLM)

### Installation

```bash
# Clone the repository
git clone https://github.com/ManoDek02/Projet-NLP.git
cd Projet-NLP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Data Preparation

```bash
# Prepare raw data
python scripts/prepare_data.py

# Index conversations into vector store
python scripts/index_conversations.py
```

### Run the Application

```bash
# Start API server
python run_api.py

# In another terminal, start UI
python run_frontend.py
```

Access:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- ** UI**: http://localhost:3000

## Project Structure

```
reddit-rag-chatbot/
├── api/                    # FastAPI REST API
│   ├── main.py            # Application setup
│   ├── routes/            # API endpoints
│   └── schemas/           # Request/Response models
├── src/
│   ├── config/            # Configuration management
│   ├── core/              # RAG components
│   │   ├── embeddings.py  # Sentence-transformers
│   │   ├── vector_store.py # ChromaDB
│   │   └── llm_handler.py # LLM integration
│   ├── services/          # Business logic
│   ├── models/            # Pydantic schemas
│   └── utils/             # Utilities
├── ui/                    # User interfaces
│   ├── gradio_app.py     # Gradio interface
│   └── streamlit_app.py  # Streamlit interface
├── scripts/               # Data processing scripts
├── tests/                 # Test suite
├── docker/                # Docker configuration
└── data/                  # Data directory
```

## Configuration

All configuration is managed via environment variables. See `.env.example` for all options.

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM backend (ollama/openai/anthropic) | `ollama` |
| `LLM_MODEL` | Model name | `llama3.2` |
| `EMBEDDING_MODEL` | Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` |
| `API_PORT` | API server port | `8000` |
| `UI_PORT` | UI server port | `3000` |

## API Usage

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What do you think about AI?",
    "use_llm": true,
    "n_results": 5
  }'
```

### Health Check

```bash
curl http://localhost:8000/api/v1/health/
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f
```

## Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check .
ruff format .

# Type checking
mypy src/
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration
```

## Architecture

### RAG Pipeline

```
User Query
    ↓
Input Validation & Preprocessing
    ↓
Text Embedding (sentence-transformers)
    ↓
Vector Similarity Search (ChromaDB)
    ↓
Context Retrieval (top-k results)
    ↓
Response Generation
    ├─→ Simple Mode: Best match response
    └─→ LLM Mode: Generated with context
    ↓
Response with Sources & Metadata
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Embeddings | sentence-transformers | Text vectorization |
| Vector Store | ChromaDB | Similarity search |
| LLM | Ollama/OpenAI/Anthropic | Response generation |
| API | FastAPI | REST interface |
| UI | HTML/CSS | Web interface |

## Performance

- **Search Latency**: < 100ms
- **Dataset Size**: 56,295 conversations
- **Embedding Dimension**: 384
- **Supported Languages**: 60+

## Benchmarking

```bash
python scripts/benchmark.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [sentence-transformers](https://www.sbert.net/) for multilingual embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Gradio](https://gradio.app/) for the web interface
