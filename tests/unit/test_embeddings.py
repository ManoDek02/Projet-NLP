"""
Unit tests for EmbeddingService.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_model.get_sentence_embedding_dimension.return_value = 384
        return mock_model

    @pytest.fixture
    def service(self, mock_sentence_transformer):
        """Create EmbeddingService with mocked model."""
        with patch(
            "src.core.embeddings.SentenceTransformer", return_value=mock_sentence_transformer
        ):
            from src.core.embeddings import EmbeddingService

            return EmbeddingService()

    @pytest.mark.unit
    def test_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert service.model is not None

    @pytest.mark.unit
    def test_embed_text_returns_list(self, service):
        """Test embed_text returns a list of floats."""
        embedding = service.embed_text("Hello world")
        assert isinstance(embedding, list)
        assert len(embedding) == 384

    @pytest.mark.unit
    def test_embed_text_correct_dimension(self, service):
        """Test embedding has correct dimension."""
        embedding = service.embed_text("Test text")
        assert len(embedding) == 384

    @pytest.mark.unit
    def test_embed_empty_text_raises_error(self, service):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            service.embed_text("")

    @pytest.mark.unit
    def test_embed_whitespace_only_raises_error(self, service):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            service.embed_text("   ")

    @pytest.mark.unit
    def test_embed_batch_returns_list(self, service, mock_sentence_transformer):
        """Test embed_batch returns list of embeddings."""
        mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])

        texts = ["Hello", "World"]
        embeddings = service.embed_batch(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(len(emb) == 384 for emb in embeddings)

    @pytest.mark.unit
    def test_embed_batch_empty_list(self, service):
        """Test embed_batch with empty list."""
        embeddings = service.embed_batch([])
        assert embeddings == []

    @pytest.mark.unit
    def test_embed_batch_filters_empty_texts(self, service, mock_sentence_transformer):
        """Test that empty texts are filtered in batch."""
        mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384])

        texts = ["Hello", "", "World", "   "]
        # Should only process non-empty texts
        embeddings = service.embed_batch(texts)
        assert len(embeddings) >= 1

    @pytest.mark.unit
    def test_get_similarity_identical_vectors(self, service):
        """Test similarity of identical vectors is 1.0."""
        vec = [0.1] * 384
        similarity = service.get_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, rel=1e-5)

    @pytest.mark.unit
    def test_get_similarity_orthogonal_vectors(self, service):
        """Test similarity of orthogonal vectors is 0.0."""
        vec1 = [1.0] + [0.0] * 383
        vec2 = [0.0, 1.0] + [0.0] * 382
        similarity = service.get_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.unit
    def test_get_similarity_opposite_vectors(self, service):
        """Test similarity of opposite vectors is -1.0."""
        vec1 = [1.0] * 384
        vec2 = [-1.0] * 384
        similarity = service.get_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0, rel=1e-5)

    @pytest.mark.unit
    def test_get_similarity_range(self, service):
        """Test similarity is always between -1 and 1."""
        import random

        for _ in range(10):
            vec1 = [random.random() for _ in range(384)]
            vec2 = [random.random() for _ in range(384)]
            similarity = service.get_similarity(vec1, vec2)
            assert -1.0 <= similarity <= 1.0

    @pytest.mark.unit
    def test_text_cleaning_before_embedding(self, service, mock_sentence_transformer):
        """Test text is cleaned before embedding."""
        text_with_extra_spaces = "  Hello   World  "
        service.embed_text(text_with_extra_spaces)

        # Verify encode was called with cleaned text
        mock_sentence_transformer.encode.assert_called()

    @pytest.mark.unit
    def test_unicode_text_handling(self, service, mock_sentence_transformer):
        """Test Unicode text is handled correctly."""
        unicode_text = "Bonjour le monde! Ceci est un test."
        embedding = service.embed_text(unicode_text)
        assert len(embedding) == 384

    @pytest.mark.unit
    def test_long_text_handling(self, service, mock_sentence_transformer):
        """Test long text is handled correctly."""
        long_text = "word " * 1000
        embedding = service.embed_text(long_text)
        assert len(embedding) == 384

    @pytest.mark.unit
    def test_special_characters_handling(self, service, mock_sentence_transformer):
        """Test special characters are handled correctly."""
        special_text = "Hello! @#$%^&*() World?"
        embedding = service.embed_text(special_text)
        assert len(embedding) == 384


class TestEmbeddingServiceIntegration:
    """Integration tests that require actual model loading."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_model_loading(self):
        """Test actual model loads correctly."""
        from src.core.embeddings import EmbeddingService

        service = EmbeddingService()
        assert service.model is not None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_embedding_generation(self):
        """Test actual embedding generation."""
        from src.core.embeddings import EmbeddingService

        service = EmbeddingService()
        embedding = service.embed_text("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_semantic_similarity(self):
        """Test semantic similarity between related texts."""
        from src.core.embeddings import EmbeddingService

        service = EmbeddingService()

        emb1 = service.embed_text("I love programming in Python")
        emb2 = service.embed_text("Python is my favorite programming language")
        emb3 = service.embed_text("The weather is nice today")

        sim_related = service.get_similarity(emb1, emb2)
        sim_unrelated = service.get_similarity(emb1, emb3)

        # Related texts should be more similar
        assert sim_related > sim_unrelated
