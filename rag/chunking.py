import re
from typing import List


def split_on_blank_lines(text: str) -> List[str]:
    """Split text into paragraphs, using blank lines as separators."""
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n+", text.strip())
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


def split_overlong_paragraph(paragraph: str, max_chars: int) -> List[str]:
    """Split a single paragraph into smaller blocks that fit inside the limit."""
    if len(paragraph) <= max_chars:
        return [paragraph]

    words = paragraph.split()
    pieces: List[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = word

    if current:
        pieces.append(current)

    return pieces


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into paragraph-based chunks.

    - blank lines define paragraph boundaries
    - paragraphs are packed into chunks until the character limit is reached
    - the last `overlap` characters of the previous chunk are prepended to the next chunk
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_chars:
        overlap = max_chars - 1

    paragraphs = []
    for paragraph in split_on_blank_lines(text):
        paragraphs.extend(split_overlong_paragraph(paragraph, max_chars))

    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue

        candidate = f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        chunks.append(current)

        if overlap and len(current) > overlap:
            carry = current[-overlap:]
            max_carry = max(0, max_chars - len(paragraph) - 2)
            if max_carry < len(carry):
                carry = carry[-max_carry:] if max_carry else ""
            current = f"{carry}\n\n{paragraph}" if carry else paragraph
        else:
            current = paragraph

    if current:
        chunks.append(current)

    return chunks
