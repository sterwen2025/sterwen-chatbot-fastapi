"""
Embedding Service using Google's gemini-embedding-001

This module handles text embedding generation for RAG functionality.
Uses Google's gemini-embedding-001 model for semantic search on meeting notes.
"""

from google import genai
from google.genai import types
import os
from typing import List, Dict
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingService:
    """Service for generating embeddings using Google's gemini-embedding-001"""

    def __init__(self, api_key: str):
        """
        Initialize the embedding service

        Args:
            api_key: Google Gemini API key
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-embedding-001"

    def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed
            task_type: Either "RETRIEVAL_DOCUMENT" (for indexing) or "RETRIEVAL_QUERY" (for search)

        Returns:
            List of 768 floats representing the embedding
        """
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768  # Set to 768 for MongoDB Atlas vector search
                )
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise

    def generate_embeddings_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 100,
        delay: float = 1.0
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with rate limiting

        Args:
            texts: List of texts to embed
            task_type: Either "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
            batch_size: Number of texts to process before delay
            delay: Seconds to wait between batches (to avoid rate limits)

        Returns:
            List of embeddings (each embedding is a list of 768 floats)
        """
        embeddings = []
        total = len(texts)

        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text, task_type)
                embeddings.append(embedding)

                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"  Embedded {i + 1}/{total} texts...")

                # Rate limiting: delay after every batch_size items
                if (i + 1) % batch_size == 0 and i + 1 < total:
                    print(f"  Rate limit pause ({delay}s)...")
                    time.sleep(delay)

            except Exception as e:
                print(f"  Error embedding text {i + 1}: {str(e)}")
                # Return None for failed embeddings
                embeddings.append(None)

        print(f"  Completed: {total} embeddings generated")
        return embeddings


def chunk_meeting_note(
    meeting_doc: Dict,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:
    """
    Split a meeting note into semantic chunks using recursive chunking

    Strategy:
    - Use RecursiveCharacterTextSplitter for intelligent chunking
    - Splits at natural boundaries: paragraphs → sentences → words → chars
    - Overlap chunks to preserve context
    - Include conclusion as separate chunk
    - Attach metadata to each chunk

    Args:
        meeting_doc: Meeting document from MongoDB
        chunk_size: Target size for each chunk (in characters)
        overlap: Number of characters to overlap between chunks

    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunks = []

    # Extract metadata (will be attached to all chunks)
    metadata = {
        "meeting_id": meeting_doc.get("ID", ""),
        "fund_name": meeting_doc.get("UniqueName", ""),
        "manager": meeting_doc.get("manager", ""),
        "contact_person": meeting_doc.get("contact_person", ""),
        "meeting_date": meeting_doc.get("meeting_date"),
        "strategy": meeting_doc.get("investment_strategy", ""),
        "importance": meeting_doc.get("importance", "")  # String field: "Selected Manager", "Monitored Manager", "#N/A"
    }

    # Get meeting notes and conclusion
    meeting_notes = meeting_doc.get("meeting_notes", "")
    conclusion = meeting_doc.get("conclusion", "")

    # Ensure meeting_notes is a string (handle NaN/float values)
    if not isinstance(meeting_notes, str):
        meeting_notes = ""

    # Ensure conclusion is a string (handle NaN/float values)
    if not isinstance(conclusion, str):
        conclusion = ""

    # Initialize recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try paragraphs, lines, sentences, words, chars
    )

    # Strategy 1: Chunk the meeting notes with recursive splitting
    if meeting_notes:
        text_chunks = text_splitter.split_text(meeting_notes)

        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_type = "full_notes" if len(text_chunks) == 1 else "notes_part"
            chunks.append({
                "text": chunk_text.strip(),
                "chunk_type": chunk_type,
                "chunk_index": chunk_index,
                **metadata
            })

    # Strategy 2: Always include conclusion as separate chunk (if exists)
    if conclusion and conclusion.strip():
        chunks.append({
            "text": conclusion.strip(),
            "chunk_type": "conclusion",
            "chunk_index": len(chunks),
            **metadata
        })

    return chunks


def prepare_chunk_for_storage(chunk: Dict, embedding: List[float]) -> Dict:
    """
    Prepare a chunk document for MongoDB storage

    Args:
        chunk: Chunk dictionary from chunk_meeting_note()
        embedding: Embedding vector (768 floats)

    Returns:
        Document ready for MongoDB insertion
    """
    from datetime import datetime

    chunk_id = f"{chunk['meeting_id']}-chunk-{chunk['chunk_index']}"

    return {
        "chunk_id": chunk_id,
        "meeting_id": chunk["meeting_id"],
        "chunk_index": chunk["chunk_index"],
        "chunk_type": chunk["chunk_type"],

        # Content
        "text": chunk["text"],
        "embedding": embedding,

        # Metadata for filtering
        "fund_name": chunk["fund_name"],
        "manager": chunk["manager"],
        "contact_person": chunk["contact_person"],
        "meeting_date": chunk["meeting_date"],
        "strategy": chunk["strategy"],
        "importance": chunk["importance"],

        # System metadata
        "embedding_model": "gemini-embedding-001",
        "created_at": datetime.utcnow()
    }
