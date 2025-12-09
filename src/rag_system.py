"""
RAG (Retrieval-Augmented Generation) module for NYC Rat Risk Intelligence Platform.

This module implements semantic search over historical 311 complaints
and NYC Health Department guidelines using sentence embeddings and ChromaDB.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

from . import config

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for sentence embedding model."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of sentence-transformers model
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Loaded embedding model: {self.model_name}")
        
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed([text])[0]


class ComplaintRAG:
    """RAG system for 311 rat complaints."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        collection_name: str = None,
        persist_directory: str = None,
    ):
        """
        Initialize the complaint RAG system.
        
        Args:
            embedding_model: Embedding model instance
            collection_name: Name of ChromaDB collection
            persist_directory: Directory to persist ChromaDB
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
        
        # Initialize ChromaDB (new API)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
        
    def build_index(
        self,
        complaints_df: pd.DataFrame,
        text_column: str = "descriptor",
        metadata_columns: List[str] = None,
        batch_size: int = 1000,
    ):
        """
        Build the complaint index from DataFrame.
        
        Args:
            complaints_df: DataFrame with complaint data
            text_column: Column containing text to embed
            metadata_columns: Columns to include as metadata
            batch_size: Batch size for embedding
        """
        logger.info(f"Building index from {len(complaints_df)} complaints...")
        
        metadata_columns = metadata_columns or [
            "created_date", "borough", "zip_code", "status",
            "resolution_description", "latitude", "longitude",
        ]
        
        # Filter to available columns
        available_metadata = [c for c in metadata_columns if c in complaints_df.columns]
        
        # Process in batches
        for start_idx in range(0, len(complaints_df), batch_size):
            end_idx = min(start_idx + batch_size, len(complaints_df))
            batch_df = complaints_df.iloc[start_idx:end_idx]
            
            # Get texts
            texts = batch_df[text_column].fillna("").astype(str).tolist()
            
            # Generate embeddings
            embeddings = self.embedding_model.embed(texts)
            
            # Prepare metadata
            metadatas = []
            for _, row in batch_df.iterrows():
                metadata = {}
                for col in available_metadata:
                    val = row.get(col, "")
                    # Convert to string for ChromaDB
                    if pd.isna(val):
                        val = ""
                    elif isinstance(val, (pd.Timestamp, np.datetime64)):
                        val = str(val)
                    else:
                        val = str(val)
                    metadata[col] = val
                metadatas.append(metadata)
                
            # Generate IDs
            ids = [f"complaint_{start_idx + i}" for i in range(len(texts))]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            
            logger.info(f"Indexed {end_idx}/{len(complaints_df)} complaints")
            
        logger.info("Index built successfully")
        
    def search(
        self,
        query: str,
        top_k: int = None,
        where: Dict = None,
        where_document: Dict = None,
    ) -> List[Dict]:
        """
        Search for similar complaints.
        
        Args:
            query: Search query
            top_k: Number of results to return
            where: Metadata filter (e.g., {"borough": "Manhattan"})
            where_document: Document content filter
            
        Returns:
            List of result dictionaries
        """
        top_k = top_k or config.RAG_TOP_K
        
        # Embed query
        query_embedding = self.embedding_model.embed_single(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            where_document=where_document,
        )
        
        # Format results
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                result = {
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "id": results["ids"][0][i] if results["ids"] else None,
                }
                formatted_results.append(result)
                
        return formatted_results
    
    def search_by_location(
        self,
        zip_code: str = None,
        borough: str = None,
        top_k: int = None,
    ) -> List[Dict]:
        """
        Search complaints by location.
        
        Args:
            zip_code: ZIP code to filter by
            borough: Borough to filter by
            top_k: Number of results
            
        Returns:
            List of complaint dictionaries
        """
        where = {}
        if zip_code:
            where["zip_code"] = zip_code
        if borough:
            where["borough"] = borough
            
        # Use a generic query for location-based search
        query = "rat sighting rodent complaint"
        
        return self.search(query, top_k=top_k, where=where if where else None)
    
    def get_location_summary(
        self,
        zip_code: str = None,
        borough: str = None,
    ) -> Dict:
        """
        Get a summary of complaints for a location.
        
        Args:
            zip_code: ZIP code
            borough: Borough
            
        Returns:
            Summary dictionary
        """
        results = self.search_by_location(zip_code, borough, top_k=100)
        
        if not results:
            return {
                "total_complaints": 0,
                "recent_complaints": [],
                "common_issues": [],
            }
            
        # Extract dates and count
        dates = []
        issues = []
        
        for r in results:
            if r["metadata"].get("created_date"):
                dates.append(r["metadata"]["created_date"])
            issues.append(r["text"])
            
        return {
            "total_complaints": len(results),
            "recent_complaints": results[:5],
            "sample_issues": issues[:10],
        }


class GuidelinesRAG:
    """RAG system for health guidelines."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        collection_name: str = "health_guidelines",
        persist_directory: str = None,
    ):
        """
        Initialize the guidelines RAG system.
        
        Args:
            embedding_model: Embedding model instance
            collection_name: Name of ChromaDB collection
            persist_directory: Directory to persist ChromaDB
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
        
        # Initialize ChromaDB (new API)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
    def build_index(
        self,
        documents_dir: Path = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Build index from guideline documents.
        
        Args:
            documents_dir: Directory containing documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        documents_dir = documents_dir or config.RAW_DATA_DIR / "health_guidelines"
        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        logger.info(f"Building guidelines index from {documents_dir}...")
        
        all_chunks = []
        all_metadatas = []
        
        # Process each document
        for doc_path in documents_dir.glob("*.txt"):
            with open(doc_path, "r") as f:
                content = f.read()
                
            # Split into chunks
            chunks = self._chunk_text(content, chunk_size, chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": doc_path.name,
                    "chunk_index": i,
                })
                
        if not all_chunks:
            logger.warning("No documents found to index")
            return
            
        # Generate embeddings
        embeddings = self.embedding_model.embed(all_chunks)
        
        # Generate IDs
        ids = [f"guideline_{i}" for i in range(len(all_chunks))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=ids,
        )
        
        # Persist
        self.client.persist()
        logger.info(f"Indexed {len(all_chunks)} guideline chunks")
        
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> List[str]:
        """Split text into overlapping chunks."""
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def search(
        self,
        query: str,
        top_k: int = None,
    ) -> List[Dict]:
        """
        Search for relevant guidelines.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of result dictionaries
        """
        top_k = top_k or config.RAG_TOP_K
        
        # Embed query
        query_embedding = self.embedding_model.embed_single(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        
        # Format results
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                result = {
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                }
                formatted_results.append(result)
                
        return formatted_results


class RAGSystem:
    """Combined RAG system for complaints and guidelines."""
    
    def __init__(self):
        """Initialize the combined RAG system."""
        self.embedding_model = EmbeddingModel()
        self.complaint_rag = ComplaintRAG(self.embedding_model)
        self.guidelines_rag = GuidelinesRAG(self.embedding_model)
        
    def build_all_indexes(self, complaints_df: pd.DataFrame = None):
        """
        Build all indexes.
        
        Args:
            complaints_df: DataFrame with complaint data
        """
        if complaints_df is not None:
            self.complaint_rag.build_index(complaints_df)
            
        self.guidelines_rag.build_index()
        
    def get_context_for_query(
        self,
        query: str,
        zip_code: str = None,
        include_complaints: bool = True,
        include_guidelines: bool = True,
    ) -> str:
        """
        Get relevant context for a query.
        
        Args:
            query: User query
            zip_code: Location to filter complaints
            include_complaints: Whether to include complaint history
            include_guidelines: Whether to include guidelines
            
        Returns:
            Combined context string
        """
        context_parts = []
        
        # Get relevant complaints
        if include_complaints:
            if zip_code:
                complaints = self.complaint_rag.search_by_location(
                    zip_code=zip_code, top_k=5
                )
            else:
                complaints = self.complaint_rag.search(query, top_k=5)
                
            if complaints:
                context_parts.append("## Relevant Historical Complaints\n")
                for i, c in enumerate(complaints, 1):
                    context_parts.append(
                        f"{i}. {c['text']} "
                        f"(Location: {c['metadata'].get('zip_code', 'N/A')}, "
                        f"Date: {c['metadata'].get('created_date', 'N/A')})\n"
                    )
                    
        # Get relevant guidelines
        if include_guidelines:
            guidelines = self.guidelines_rag.search(query, top_k=3)
            
            if guidelines:
                context_parts.append("\n## Relevant Health Guidelines\n")
                for g in guidelines:
                    context_parts.append(f"{g['text']}\n\n")
                    
        return "".join(context_parts)
    
    def answer_question(
        self,
        question: str,
        zip_code: str = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Get context and sources for a question.
        
        Args:
            question: User question
            zip_code: Location context
            
        Returns:
            Tuple of (context, sources)
        """
        context = self.get_context_for_query(
            question,
            zip_code=zip_code,
        )
        
        # Get sources
        sources = []
        sources.extend(self.complaint_rag.search(question, top_k=3))
        sources.extend(self.guidelines_rag.search(question, top_k=2))
        
        return context, sources


def build_rag_index(complaints_df: pd.DataFrame = None):
    """
    Build the complete RAG index.
    
    Args:
        complaints_df: DataFrame with complaint data
    """
    logger.info("Building RAG index...")
    
    rag = RAGSystem()
    
    if complaints_df is None:
        # Load from processed data
        processed_path = config.RAW_DATA_DIR / "rat_sightings.csv"
        if processed_path.exists():
            complaints_df = pd.read_csv(processed_path, low_memory=False)
            
    rag.build_all_indexes(complaints_df)
    
    logger.info("RAG index complete")
    
    return rag


if __name__ == "__main__":
    # Test RAG system
    logger.info("Testing RAG system...")
    
    # Create embedding model
    embed_model = EmbeddingModel()
    
    # Test embedding
    test_texts = [
        "Rat sighting in alley behind restaurant",
        "Mouse droppings found in basement",
        "Rodent burrow near garbage cans",
    ]
    
    embeddings = embed_model.embed(test_texts)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Test guidelines RAG
    guidelines_rag = GuidelinesRAG(embed_model)
    guidelines_rag.build_index()
    
    results = guidelines_rag.search("how to prevent rats")
    print(f"\nGuidelines search results: {len(results)}")
    for r in results[:2]:
        print(f"- {r['text'][:100]}...")
