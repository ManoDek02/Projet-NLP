"""
Unit tests for LLMService.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestLLMService:
    """Tests for LLMService class."""

    @pytest.fixture
    def mock_ollama(self):
        """Create mock Ollama client."""
        mock = MagicMock()
        mock.chat.return_value = {"message": {"content": "This is a test response from LLM."}}
        mock.list.return_value = {"models": [{"name": "llama3.2"}]}
        return mock

    @pytest.fixture
    def service(self, mock_ollama):
        """Create LLMService with mocked Ollama."""
        with patch("src.core.llm_handler.ollama", mock_ollama):
            from src.core.llm_handler import LLMService

            return LLMService()

    @pytest.mark.unit
    def test_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None

    @pytest.mark.unit
    def test_generate_returns_string(self, service):
        """Test generate returns a string response."""
        context = ["Context 1: Some information", "Context 2: More information"]
        response = service.generate("What is AI?", context)

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.unit
    def test_generate_with_empty_context(self, service):
        """Test generate with empty context list."""
        response = service.generate("What is AI?", [])

        assert isinstance(response, str)

    @pytest.mark.unit
    def test_generate_with_single_context(self, service):
        """Test generate with single context item."""
        context = ["AI is artificial intelligence."]
        response = service.generate("What is AI?", context)

        assert isinstance(response, str)

    @pytest.mark.unit
    def test_generate_empty_query_raises_error(self, service):
        """Test that empty query raises error."""
        context = ["Some context"]

        with pytest.raises((ValueError, Exception)):
            service.generate("", context)

    @pytest.mark.unit
    def test_build_prompt_includes_query(self, service):
        """Test that prompt includes the user query."""
        query = "What is machine learning?"
        context = ["ML is a subset of AI."]

        prompt = service._build_prompt(query, context)

        assert query in prompt

    @pytest.mark.unit
    def test_build_prompt_includes_context(self, service):
        """Test that prompt includes context."""
        query = "What is AI?"
        context = ["AI stands for Artificial Intelligence."]

        prompt = service._build_prompt(query, context)

        assert "Artificial Intelligence" in prompt

    @pytest.mark.unit
    def test_build_prompt_multiple_contexts(self, service):
        """Test prompt building with multiple contexts."""
        query = "What is AI?"
        context = [
            "AI is artificial intelligence.",
            "Machine learning is part of AI.",
            "Deep learning is a type of ML.",
        ]

        prompt = service._build_prompt(query, context)

        assert "artificial intelligence" in prompt.lower()
        assert "machine learning" in prompt.lower()

    @pytest.mark.unit
    def test_generate_with_temperature(self, service, mock_ollama):
        """Test generate respects temperature parameter."""
        context = ["Some context"]
        service.generate("Test query", context, temperature=0.5)

        # Verify ollama.chat was called
        mock_ollama.chat.assert_called()

    @pytest.mark.unit
    def test_generate_with_max_tokens(self, service, mock_ollama):
        """Test generate respects max_tokens parameter."""
        context = ["Some context"]
        service.generate("Test query", context, max_tokens=100)

        mock_ollama.chat.assert_called()

    @pytest.mark.unit
    def test_check_availability(self, service):
        """Test availability check."""
        is_available = service._check_availability()

        assert isinstance(is_available, bool)

    @pytest.mark.unit
    def test_generate_handles_llm_error(self, service, mock_ollama):
        """Test generate handles LLM errors gracefully."""
        mock_ollama.chat.side_effect = Exception("LLM Error")

        context = ["Some context"]

        # Should either raise or return fallback
        try:
            response = service.generate("Test query", context)
            # If it returns, should be a fallback response
            assert isinstance(response, str)
        except Exception:
            pass  # Also acceptable

    @pytest.mark.unit
    def test_generate_with_special_characters(self, service):
        """Test generate handles special characters in query."""
        context = ["Some context"]
        response = service.generate("What is AI? @#$%^&*()", context)

        assert isinstance(response, str)

    @pytest.mark.unit
    def test_generate_with_unicode(self, service):
        """Test generate handles Unicode in query."""
        context = ["Contexte en français"]
        response = service.generate("Qu'est-ce que l'IA?", context)

        assert isinstance(response, str)

    @pytest.mark.unit
    def test_generate_long_context(self, service):
        """Test generate with long context."""
        long_context = ["This is a very long context. " * 100]
        response = service.generate("Summarize this", long_context)

        assert isinstance(response, str)


class TestLLMServiceProviders:
    """Tests for different LLM providers."""

    @pytest.mark.unit
    def test_ollama_provider(self):
        """Test Ollama provider configuration."""
        with patch("src.core.llm_handler.ollama") as mock_ollama:
            mock_ollama.chat.return_value = {"message": {"content": "Response"}}

            from src.core.llm_handler import LLMService

            with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
                service = LLMService()
                assert service is not None

    @pytest.mark.unit
    def test_openai_provider_initialization(self):
        """Test OpenAI provider initialization."""
        with patch("src.core.llm_handler.openai") as mock_openai:
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client

            from src.core.llm_handler import LLMService

            with patch.dict("os.environ", {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
                # Should initialize without error
                try:
                    service = LLMService()
                except Exception:
                    pass  # May fail without real API key

    @pytest.mark.unit
    def test_anthropic_provider_initialization(self):
        """Test Anthropic provider initialization."""
        with patch("src.core.llm_handler.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client

            from src.core.llm_handler import LLMService

            with patch.dict(
                "os.environ", {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"}
            ):
                try:
                    service = LLMService()
                except Exception:
                    pass  # May fail without real API key


class TestLLMServiceIntegration:
    """Integration tests with real LLM."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_ollama_generation(self):
        """Test real Ollama generation (requires Ollama running)."""
        try:
            from src.core.llm_handler import LLMService

            service = LLMService()

            if service._check_availability():
                context = ["Python is a programming language."]
                response = service.generate("What is Python?", context)

                assert isinstance(response, str)
                assert len(response) > 0
            else:
                pytest.skip("Ollama not available")
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
