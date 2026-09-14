"""Offline tests for the chunker: no API key, no model downloads."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunking import chunk_text


MAX_CHARS = 200

# 30 paragraphs of ordinary prose -> comfortably longer than one chunk.
LONG_DOCUMENT = "\n\n".join(
    f"Paragraph {i} talks about retrieval augmented generation "
    f"and why chunk boundaries matter for recall."
    for i in range(30)
)


def test_long_document_splits_into_bounded_chunks():
    chunks = chunk_text(LONG_DOCUMENT, max_chars=MAX_CHARS, overlap=50)

    assert len(chunks) > 1, "a long document should not collapse into a single chunk"

    oversized = [(i, len(c)) for i, c in enumerate(chunks) if len(c) > MAX_CHARS]
    assert not oversized, f"chunks exceeded max_chars={MAX_CHARS}: {oversized}"
