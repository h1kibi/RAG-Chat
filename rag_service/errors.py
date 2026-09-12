class RagError(RuntimeError):
    """Base class for standalone retrieval failures."""


class RagIndexNotReadyError(RagError):
    """The requested knowledge base has no usable matching vector index."""


class RagEmbeddingError(RagError):
    """The configured standalone embedding provider could not embed a query."""
