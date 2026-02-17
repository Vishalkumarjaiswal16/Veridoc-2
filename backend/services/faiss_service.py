import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FAISS_INDEX_DIR, EMBEDDINGS_DIMENSION

logger = logging.getLogger(__name__)


class FAISSService:
    def __init__(self, index_dir: str = None, dimension: int = None):
        """Initialize FAISS index for high-performance search"""
        self.index_dir = Path(index_dir or FAISS_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension or EMBEDDINGS_DIMENSION
        
        # Try to load existing index, or create new one
        if not self.load_index():
            # Create HNSW index (better than flat L2 for similarity search)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 connections per node
            self.index.hnsw.efConstruction = 200
            self.id_map = {}  # Map FAISS index to document IDs
            self.metadata_store = {}  # Store metadata for retrieved docs
            self.document_store = {}  # Store document content
        
        logger.info(f"FAISS index initialized with dimension {self.dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, 
                      ids: List[str], 
                      metadata: List[Dict],
                      documents: Optional[List[str]] = None) -> bool:
        """
        Add embeddings to FAISS index.
        """
        try:
            # Ensure embeddings are float32
            embeddings = embeddings.astype(np.float32)
            
            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(embeddings)
            
            # Map indices to IDs and store metadata
            for i, doc_id in enumerate(ids):
                self.id_map[start_idx + i] = doc_id
                self.metadata_store[doc_id] = metadata[i]
                if documents:
                    self.document_store[doc_id] = documents[i]
            
            logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {str(e)}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict]:
        """
        Search FAISS index using query embedding.
        Returns top_k most similar documents.
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            
            # Adjust top_k if larger than index size
            top_k = min(top_k, self.index.ntotal)
            
            # Search index
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # -1 indicates empty result
                    doc_id = self.id_map.get(idx)
                    if doc_id:
                        result = {
                            'id': doc_id,
                            'distance': float(distances[0][i]),
                            'similarity_score': 1.0 / (1.0 + float(distances[0][i])),
                            'metadata': self.metadata_store.get(doc_id, {})
                        }
                        # Add document content if available
                        if doc_id in self.document_store:
                            result['content'] = self.document_store[doc_id]
                        retrieved_docs.append(result)
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in index"""
        return self.index.ntotal
    
    def save_index(self) -> bool:
        """Save FAISS index and metadata to disk"""
        try:
            index_path = self.index_dir / "index.faiss"
            metadata_path = self.index_dir / "metadata.json"
            id_map_path = self.index_dir / "id_map.json"
            documents_path = self.index_dir / "documents.json"
            
            faiss.write_index(self.index, str(index_path))
            
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata_store, f)
            
            with open(id_map_path, 'w') as f:
                json.dump({str(k): v for k, v in self.id_map.items()}, f)
            
            with open(documents_path, 'w') as f:
                json.dump(self.document_store, f)
            
            logger.info("FAISS index saved to disk")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            return False
    
    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            index_path = self.index_dir / "index.faiss"
            metadata_path = self.index_dir / "metadata.json"
            id_map_path = self.index_dir / "id_map.json"
            documents_path = self.index_dir / "documents.json"
            
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                
                with open(metadata_path, 'r') as f:
                    self.metadata_store = json.load(f)
                
                with open(id_map_path, 'r') as f:
                    id_map_json = json.load(f)
                    self.id_map = {int(k): v for k, v in id_map_json.items()}
                
                if documents_path.exists():
                    with open(documents_path, 'r') as f:
                        self.document_store = json.load(f)
                else:
                    self.document_store = {}
                
                logger.info("FAISS index loaded from disk")
                return True
            else:
                logger.warning("No existing FAISS index found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return False
    
    def reset_index(self) -> bool:
        """Reset the FAISS index (for testing)"""
        try:
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.id_map = {}
            self.metadata_store = {}
            self.document_store = {}
            logger.info("FAISS index reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset index: {str(e)}")
            return False


# Initialize FAISS service (lazy loading)
_faiss_service = None


def get_faiss_service() -> FAISSService:
    """Get or create FAISS service singleton."""
    global _faiss_service
    if _faiss_service is None:
        _faiss_service = FAISSService()
    return _faiss_service


# For backwards compatibility
faiss_service = None


def init_faiss_service():
    """Initialize the global FAISS service."""
    global faiss_service
    faiss_service = get_faiss_service()
    return faiss_service
