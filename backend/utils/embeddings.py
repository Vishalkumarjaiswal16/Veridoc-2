"""
Embeddings Manager Module for VeriDoc
=====================================

This module provides text embedding generation using sentence-transformers.

Model Selection Rationale
-------------------------
**Selected Model:** sentence-transformers/all-MiniLM-L6-v2

**Why this model was chosen:**

1. **Optimal Size/Performance Balance**
   - 384-dimensional embeddings (compact yet expressive)
   - ~80MB model size (fast to load, low memory footprint)
   - ~14,000 sentences/second on CPU (highly efficient)

2. **Quality Metrics**
   - Strong performance on Semantic Textual Similarity (STS) benchmarks
   - Trained on 1B+ sentence pairs from diverse sources
   - Good generalization across domains (technical docs, Q&A, general text)

3. **Production-Ready Characteristics**
   - Apache 2.0 license (commercial use permitted)
   - Widely adopted and well-maintained
   - Excellent documentation and community support

**Alternative Models Considered:**

| Model | Dimensions | Speed | Quality | Notes |
|-------|------------|-------|---------|-------|
| all-MiniLM-L6-v2 (SELECTED) | 384 | ★★★★★ | ★★★★☆ | Best balance |
| all-mpnet-base-v2 | 768 | ★★★☆☆ | ★★★★★ | Higher quality, slower |
| multi-qa-MiniLM-L6-cos-v1 | 384 | ★★★★★ | ★★★★☆ | Q&A optimized |
| paraphrase-MiniLM-L6-v2 | 384 | ★★★★★ | ★★★☆☆ | Paraphrase focused |

**When to consider alternatives:**
- Use `all-mpnet-base-v2` if quality is paramount and latency is acceptable
- Use `multi-qa-MiniLM-L6-cos-v1` for pure Q&A systems
- Use `multilingual-e5-base` for multi-language support
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDINGS_MODEL, EMBEDDINGS_DIMENSION
import logging

logger = logging.getLogger(__name__)


class EmbeddingsManager: # main class for managing embeddings 
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDINGS_MODEL)
        self.dimension = EMBEDDINGS_DIMENSION
        logger.info(f"Loaded embeddings model: {EMBEDDINGS_MODEL}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray: #main function for embedding for multiple documents 
        """
        Generate embeddings for list of texts.
        Handles batch processing for efficiency.
        """
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32
            )
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray: # embedding for single query 
        """Generate embedding for single query."""
        return self.model.encode(query, convert_to_numpy=True)
    
    def similarity_search(self, query_embedding: np.ndarray, # 
                         doc_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[int]:
        """
        This function compares the query vector with every document vector, scores how similar they are using cosine similarity, sorts all documents by that score, and returns the positions of the top k most similar documents.
        """
        from scipy.spatial.distance import cosine
        
        similarities = []
        for idx, doc_embedding in enumerate(doc_embeddings):
            sim = 1 - cosine(query_embedding, doc_embedding)
            similarities.append((idx, sim))
        
        # Sort by similarity (descending) and return top_k indices
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in similarities[:top_k]]


# Initialize embeddings manager (lazy loading)
_embeddings_manager = None


def get_embeddings_manager() -> EmbeddingsManager:
    """Get or create embeddings manager singleton."""
    global _embeddings_manager
    if _embeddings_manager is None:
        _embeddings_manager = EmbeddingsManager()
    return _embeddings_manager


# For backwards compatibility
embeddings_manager = None


def init_embeddings_manager():
    """Initialize the global embeddings manager."""
    global embeddings_manager
    embeddings_manager = get_embeddings_manager()
    return embeddings_manager
