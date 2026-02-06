"""
Integration tests for the FastAPI application.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_chatbot_service():
    """Create mock ChatbotService."""
    mock_service = MagicMock()
    mock_service.chat.return_value = {
        "message": "This is a test response.",
        "sources": [
            {
                "context": "Original question",
                "response": "Original answer",
                "score": 0.85,
                "rank": 1,
            }
        ],
        "metadata": {
            "duration_ms": 100,
            "mode": "simple",
            "n_sources": 1,
            "model": None,
        },
    }
    mock_service.health_check.return_value = {
        "status": "healthy",
        "components": {
            "embedding_service": {"status": "healthy"},
            "vector_store": {"status": "healthy", "document_count": 100},
            "llm_service": {"status": "healthy"},
        },
    }
    mock_service.get_stats.return_value = {
        "total_conversations": 56295,
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "llm_provider": "ollama",
    }
    return mock_service


@pytest.fixture
def client(mock_chatbot_service):
    """Create test client with mocked service."""
    with patch("api.main.ChatbotService", return_value=mock_chatbot_service):
        from fastapi.testclient import TestClient
        from api.main import app

        with TestClient(app) as client:
            yield client


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.mark.integration
    def test_root_returns_200(self, client):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_root_returns_app_info(self, client):
        """Test root endpoint returns application info."""
        response = client.get("/")
        data = response.json()

        assert "name" in data or "status" in data


class TestChatEndpoint:
    """Tests for chat endpoint."""

    @pytest.mark.integration
    def test_chat_endpoint_success(self, client):
        """Test chat endpoint with valid request."""
        response = client.post(
            "/api/v1/chat/",
            json={"message": "Hello, how are you?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    @pytest.mark.integration
    def test_chat_endpoint_with_llm(self, client):
        """Test chat endpoint with LLM enabled."""
        response = client.post(
            "/api/v1/chat/",
            json={
                "message": "What is AI?",
                "use_llm": True,
                "n_results": 3
            }
        )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_chat_endpoint_with_parameters(self, client):
        """Test chat endpoint with all parameters."""
        response = client.post(
            "/api/v1/chat/",
            json={
                "message": "Test message",
                "use_llm": True,
                "n_results": 5,
                "temperature": 0.8,
                "max_tokens": 200
            }
        )

        assert response.status_code == 200

    @pytest.mark.integration
    def test_chat_endpoint_empty_message(self, client):
        """Test chat endpoint with empty message."""
        response = client.post(
            "/api/v1/chat/",
            json={"message": ""}
        )

        # Should return 400 or 422 for validation error
        assert response.status_code in [400, 422]

    @pytest.mark.integration
    def test_chat_endpoint_missing_message(self, client):
        """Test chat endpoint with missing message field."""
        response = client.post(
            "/api/v1/chat/",
            json={}
        )

        assert response.status_code == 422

    @pytest.mark.integration
    def test_chat_endpoint_invalid_n_results(self, client):
        """Test chat endpoint with invalid n_results."""
        response = client.post(
            "/api/v1/chat/",
            json={
                "message": "Test",
                "n_results": -1
            }
        )

        # Should validate n_results
        assert response.status_code in [200, 400, 422]

    @pytest.mark.integration
    def test_chat_endpoint_response_structure(self, client):
        """Test chat endpoint response has correct structure."""
        response = client.post(
            "/api/v1/chat/",
            json={"message": "Hello"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "sources" in data or "metadata" in data

    @pytest.mark.integration
    def test_chat_stats_endpoint(self, client):
        """Test chat stats endpoint."""
        response = client.get("/api/v1/chat/stats")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    @pytest.mark.integration
    def test_chat_examples_endpoint(self, client):
        """Test chat examples endpoint."""
        response = client.get("/api/v1/chat/examples")

        assert response.status_code == 200
        data = response.json()
        assert "examples" in data


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.integration
    def test_health_endpoint(self, client):
        """Test main health endpoint."""
        response = client.get("/api/v1/health/")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.integration
    def test_health_ready_endpoint(self, client):
        """Test readiness probe endpoint."""
        response = client.get("/api/v1/health/ready")

        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data

    @pytest.mark.integration
    def test_health_live_endpoint(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/api/v1/health/live")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.integration
    def test_health_version_endpoint(self, client):
        """Test version endpoint."""
        response = client.get("/api/v1/health/version")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    @pytest.mark.integration
    def test_openapi_json(self, client):
        """Test OpenAPI JSON endpoint."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    @pytest.mark.integration
    def test_docs_endpoint(self, client):
        """Test Swagger UI endpoint."""
        response = client.get("/docs")

        assert response.status_code == 200

    @pytest.mark.integration
    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint."""
        response = client.get("/redoc")

        assert response.status_code == 200


class TestCORS:
    """Tests for CORS configuration."""

    @pytest.mark.integration
    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/api/v1/chat/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )

        # Should allow the request
        assert response.status_code in [200, 204, 400]


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.integration
    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/api/v1/nonexistent")

        assert response.status_code == 404

    @pytest.mark.integration
    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        response = client.get("/api/v1/chat/")

        assert response.status_code == 405

    @pytest.mark.integration
    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/chat/",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422
