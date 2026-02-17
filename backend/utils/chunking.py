"""
Document Chunking Module for VeriDoc
=====================================

This module provides document chunking strategies for the RAG pipeline.

Optimal Chunk Parameters
------------------------
**Default Configuration:** chunk_size=1000, chunk_overlap=200

**Why these values were chosen:**

1. **Chunk Size = 1000 characters**
   - Fits comfortably within embedding model context (MiniLM handles 512 tokens ~2000 chars)
   - Large enough to capture meaningful semantic units (paragraphs, sections)
   - Small enough to maintain retrieval precision
   - Balances context richness with search granularity

2. **Chunk Overlap = 200 characters (20%)**
   - Preserves context continuity across chunk boundaries
   - Prevents important information from being split mid-sentence
   - 20% overlap is industry standard for semantic search
   - Higher overlap = more redundancy but better boundary handling

**Trade-offs:**

| Chunk Size | Pros | Cons |
|------------|------|------|
| Smaller (500) | Higher precision, faster search | May lose context, more chunks |
| Larger (2000) | More context per chunk | Lower precision, may exceed model limits |

**Recommendations by Document Type:**

- **Technical docs**: 1000-1500 chars (preserve code blocks)
- **Legal documents**: 800-1000 chars (precise clause retrieval)
- **General text**: 1000 chars (balanced approach)
- **Q&A pairs**: 500 chars (keep questions/answers together)

**Chunking Strategies Available:**

1. `RecursiveCharacterTextSplitter` (default): Respects semantic boundaries
2. `fixed_size_overlap`: Simple character-based splitting
3. `semantic_chunking`: Sentence-boundary aware splitting
"""

from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        
        # Recursive splitter respects document structure
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]  # Semantic boundaries
        )
    
    def chunk_document(self, text: str, doc_id: str) -> List[Tuple[str, dict]]:
        """
        Split document into chunks while preserving metadata.
        Returns: List of (chunk_text, metadata) tuples
        """
        try:
            chunks = self.splitter.split_text(text)
            
            chunked_data = [
                (chunk, {
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                })
                for idx, chunk in enumerate(chunks)
            ]
            
            logger.info(f"Document {doc_id} split into {len(chunks)} chunks")
            return chunked_data
            
        except Exception as e:
            logger.error(f"Chunking failed for {doc_id}: {str(e)}")
            raise
    
    def chunk_multiple(self, documents: List[Tuple[str, str]]) -> List[Tuple[str, dict]]:
        """
        Chunk multiple documents.
        Input: List of (document_text, document_id) tuples
        """
        all_chunks = []
        for text, doc_id in documents:
            chunks = self.chunk_document(text, doc_id)
            all_chunks.extend(chunks)
        return all_chunks


class ChunkingStrategies:
    """Different chunking approaches for specific use cases"""
    
    @staticmethod
    def fixed_size_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple fixed-size chunks with overlap"""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks
    
    @staticmethod
    def semantic_chunking(text: str, sentences_per_chunk: int = 5) -> List[str]:
        """Chunk based on sentence boundaries"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i+sentences_per_chunk])
            chunks.append(chunk)
        return chunks


# Default chunker instance
chunker = DocumentChunker()
