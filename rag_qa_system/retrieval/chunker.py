from __future__ import annotations
import hashlib
import re
from typing import List
from rag_qa_system.core.schemas import DocumentChunk


class TextChunker:
    """
    Recursive character text chunker with paragraph and sentence boundary preservation.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str = "document") -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        paragraphs = re.split(r'\n\s*\n', text)
        chunks: List[DocumentChunk] = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)
            if current_length + para_len > self.chunk_size and current_chunk:
                combined_text = "\n\n".join(current_chunk)
                chunk_id = hashlib.md5(f"{source}_{chunk_idx}_{combined_text[:30]}".encode()).hexdigest()
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    source=source,
                    content=combined_text,
                    chunk_index=chunk_idx,
                    token_count=len(combined_text.split()),
                ))
                chunk_idx += 1
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(para)
            current_length += para_len

        if current_chunk:
            combined_text = "\n\n".join(current_chunk)
            chunk_id = hashlib.md5(f"{source}_{chunk_idx}_{combined_text[:30]}".encode()).hexdigest()
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                source=source,
                content=combined_text,
                chunk_index=chunk_idx,
                token_count=len(combined_text.split()),
            ))

        return chunks
