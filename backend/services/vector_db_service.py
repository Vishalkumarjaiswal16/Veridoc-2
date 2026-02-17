"""
ChromaDB Vector Database Service
================================

This module provides an interface to ChromaDB for vector storage and retrieval.

Configuration & Persistence
---------------------------
**Storage Backend:**
- Uses local persistence via `PersistentClient`.
- Data is stored in `CHROMA_PERSIST_DIR` (default: `./vector_db/chromadb`).
- Underlying engine: SQLite (for metadata) + HNSW (for vector index).

**Key Settings:**
- `anonymized_telemetry=False`: Disables data sharing.
- `allow_reset=True`: Enables complete database wipe (useful for testing).
- `hnsw:space="cosine"`: Uses Cosine Similarity for vector comparison.
  - Distance range: [0, 2] where 0 is identical.
  - Converted to similarity score [0, 1] in results.

**Usage:**
- Use `get_chroma_service()` singleton to ensure one connection per process.
- Documents are automatically embedded if embeddings are missing (not used in this project as we pre-compute).
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)


class ChromaDBService:
    def __init__(self, persist_dir: str = None):
        """Initialize ChromaDB with persistence"""
        self.persist_dir = Path(persist_dir or CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB settings for production
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = "veridoc_documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        
        logger.info(f"ChromaDB initialized with persistence at {self.persist_dir}")
    
    def add_documents(self, chunks: List[Dict]) -> bool:
        """
        Add document chunks to vector database.
        Expected format: {
            'id': unique_id,
            'content': chunk_text,
            'metadata': {doc_id, chunk_index, ...}
        }
        """
        try:
            ids = [chunk['id'] for chunk in chunks]
            documents = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            embeddings = [chunk.get('embedding') for chunk in chunks]
            
            # Filter out None embeddings
            if all(e is not None for e in embeddings):
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings  # Pre-computed embeddings
                )
            else:
                # Let ChromaDB compute embeddings
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            logger.info(f"Added {len(chunks)} chunks to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 4) -> List[Dict]:
        """
        Search vector database using query embedding.
        Returns top_k most similar documents.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    retrieved_docs.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0,
                        'similarity_score': 1 - results['distances'][0][i] if results['distances'] else 1
                    })
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def search_by_text(self, query_text: str, top_k: int = 4) -> List[Dict]:
        """
        Search vector database using query text.
        ChromaDB will compute the embedding automatically.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    retrieved_docs.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0,
                        'similarity_score': 1 - results['distances'][0][i] if results['distances'] else 1
                    })
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in collection"""
        return self.collection.count()
    
    def delete_collection(self) -> bool:
        """Delete collection (for testing/reset)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection deleted and recreated")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            return False
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False


# Initialize ChromaDB service (lazy loading)
_chroma_service = None


def get_chroma_service() -> ChromaDBService:
    """Get or create ChromaDB service singleton."""
    global _chroma_service
    if _chroma_service is None:
        _chroma_service = ChromaDBService()
    return _chroma_service


# For backwards compatibility
chroma_service = None


def init_chroma_service():
    """Initialize the global ChromaDB service."""
    global chroma_service
    chroma_service = get_chroma_service()
    return chroma_service
