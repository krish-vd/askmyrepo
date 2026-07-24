"""Embedding model singleton + batch embedding helpers.

sentence-transformers all-MiniLM-L6-v2, 384-dim. Loaded once per process
(module-level singleton) since model load is the expensive part, not inference.
"""
from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _chunk_text(chunk):
    """Prefix source with symbol_type/name/file_path so the embedding sees the
    identifier, not just body text — improves retrieval for queries that name
    a function/class directly."""
    symbol = chunk.symbol_name or "module"
    return f"{chunk.symbol_type} {symbol} in {chunk.file_path}\n{chunk.source_code}"


def embed_chunks(chunks, batch_size=32):
    """Embed a list of CodeChunk instances in place (sets .embedding) and
    bulk_update()s them. Does not save chunks that have no source_code."""
    model = get_model()
    texts = [_chunk_text(c) for c in chunks]
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector
    from .models import CodeChunk
    CodeChunk.objects.bulk_update(chunks, ["embedding"], batch_size=batch_size)


def embed_query(text: str):
    model = get_model()
    return model.encode(text, show_progress_bar=False)
