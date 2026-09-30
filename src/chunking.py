
"""
Turns a raw text corpus into a list of clean, retrievable chunks.

Uses LangChain's RecursiveCharacterTextSplitter: it tries to split on
paragraph breaks first, then lines, then sentences/words, and only cuts
mid-text as a last resort, keeping each chunk under chunk_size characters.
"""

from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 500     # max characters per chunk
DEFAULT_CHUNK_OVERLAP = 50   # characters shared between neighbouring chunks


def read_and_split_text(
    filepath: str | Path,
    min_chars: int = 20,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Read a text file and split it into chunks of at most `chunk_size`
    characters, with `chunk_overlap` characters repeated between neighbours.
    Chunks shorter than `min_chars` are dropped.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    text = filepath.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split the RAW text first. The splitter needs the newlines to find
    # paragraph boundaries, so cleaning must happen after splitting.
    raw_chunks = splitter.split_text(text)
    chunks = [_clean(c) for c in raw_chunks]
    chunks = [c for c in chunks if len(c) >= min_chars]

    if not chunks:
        raise ValueError(f"No usable chunks found in {filepath}.")

    return chunks


def _clean(chunk: str) -> str:
    """Collapse newlines and repeated whitespace into single spaces."""
    return " ".join(chunk.split())
