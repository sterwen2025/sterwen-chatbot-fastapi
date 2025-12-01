"""
Meeting Notes & Fund Comments Chatbot API
==========================================
FastAPI backend for intelligent Q&A system with meeting notes, factsheets, and transcripts

Version: 2024-11-20 - Added Gemini optimizations for faster responses
"""

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date
from pymongo import MongoClient
from bson import Regex as BsonRegex  # For Cosmos DB compatible regex
import calendar
import os
import json
import uuid
import re
import math
from google import genai
from dotenv import load_dotenv
from embedding_service import EmbeddingService
import bm25s  # BM25S for fast pre-indexed search

# Load environment variables
load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://mongodbsterwen:Genevaboy$1204@sterwendb.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Configure Gemini client (automatically uses GEMINI_API_KEY from environment)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Global MongoDB connection pool (reuse across all requests)
mongo_client = MongoClient(
    MONGO_URI,
    maxPoolSize=50,  # Maximum connections in pool
    minPoolSize=10,  # Minimum connections to maintain
    maxIdleTimeMS=120000,  # Keep connections alive
    serverSelectionTimeoutMS=5000  # Timeout for connection selection
)

# Debug: Check if API key is loaded (don't print the key itself)
print(f"GEMINI_API_KEY loaded: {bool(GEMINI_API_KEY)}")
print(f"MongoDB connection pool initialized (maxPoolSize=50)")
print("=" * 80)
print("CONTAINER VERSION: 2024-11-13-gemini-2.5-flash-switch")
print("=" * 80)

app = FastAPI(title="Meeting Notes Chatbot API", version="1.0.0")

# Helper function to strip HTML tags from text
def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text, replacing them with appropriate markdown or whitespace."""
    if not text:
        return text
    # Replace <br> and <br/> with newlines
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    # Replace </p>, </div>, </h1-6> with double newlines for paragraph breaks
    text = re.sub(r'</(p|div|h[1-6])>', '\n\n', text, flags=re.IGNORECASE)
    # Remove all other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Clean up multiple consecutive newlines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to preload caches and indexes
@app.on_event("startup")
async def startup_event():
    """Load fund names and BM25S indexes into memory at startup"""
    load_all_fund_names()
    load_meeting_bm25s_index()
    load_factsheet_bm25s_index()

# Models
class ChatRequest(BaseModel):
    question: str
    data_sources: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    selected_funds: Optional[List[str]] = None
    conversation_history: Optional[List[dict]] = None
    model: Optional[str] = "gemini-2.5-flash"  # "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-low-thinking", "gemini-3-high-thinking"

class ChatResponse(BaseModel):
    answer: str
    sources_used: List[str]
    data_summary: dict

class FilterStats(BaseModel):
    meeting_notes_count: int
    factsheet_comments_count: int
    transcripts_count: int
    total_count: int

class ChatHistoryMessage(BaseModel):
    question: str
    answer: str
    sources: List[str]
    timestamp: str

class SaveChatRequest(BaseModel):
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message: ChatHistoryMessage

class CreateConversationRequest(BaseModel):
    session_id: Optional[str] = None
    title: Optional[str] = "New Conversation"

# Helper Functions
def get_user_identifier(request: Request, session_id: Optional[str] = None) -> str:
    """
    Get user identifier from Azure AD authentication or session ID.
    Priority: Azure AD email > Azure AD user ID > session_id parameter > generated UUID
    """
    # Try to get Azure AD user from App Service Authentication headers
    azure_user_name = request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME")  # Email/username
    azure_user_id = request.headers.get("X-MS-CLIENT-PRINCIPAL-ID")      # User ID

    # Prioritize email over user ID for better stability and user-friendliness
    if azure_user_name:
        return f"azure_{azure_user_name}"
    elif azure_user_id:
        return f"azure_{azure_user_id}"
    elif session_id:
        return f"session_{session_id}"
    else:
        # Generate new session ID for anonymous users
        return f"session_{str(uuid.uuid4())}"

def get_filtered_meetings(start_date: Optional[str], end_date: Optional[str], selected_funds: Optional[List[str]]):
    """Get meeting notes filtered by date and funds."""
    try:
        db = mongo_client["Meetings"]
        collection = db["Meeting Notes"]

        # Build query
        query_conditions = []

        # Date filter
        if start_date and end_date:
            date_condition = {
                "meeting_date": {"$gte": start_date, "$lte": end_date}
            }
            query_conditions.append(date_condition)

        # Fund filter
        if selected_funds:
            fund_condition = {"UniqueName": {"$in": selected_funds}}
            query_conditions.append(fund_condition)

        # Combine conditions
        if query_conditions:
            query = {"$and": query_conditions} if len(query_conditions) > 1 else query_conditions[0]
        else:
            query = {}

        # Get meetings
        meetings = list(collection.find(query, {
            "meeting_date": 1,
            "UniqueName": 1,
            "manager": 1,
            "meeting_notes": 1,
            "conclusion": 1,
            "_id": 0
        }))

        return meetings

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting meetings: {e}")

def get_filtered_transcripts(start_date: Optional[str], end_date: Optional[str], selected_funds: Optional[List[str]]):
    """Get transcripts filtered by date and funds."""
    try:
        db = mongo_client["Meetings"]
        collection = db["transcripts"]

        # Build query
        query_conditions = []

        # Date filter (meeting_date is datetime object)
        if start_date and end_date:
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

            date_condition = {
                "meeting_date": {"$gte": start_datetime, "$lte": end_datetime}
            }
            query_conditions.append(date_condition)

        # Fund filter
        if selected_funds:
            fund_condition = {"UniqueName": {"$in": selected_funds}}
            query_conditions.append(fund_condition)

        # Combine conditions
        if query_conditions:
            query = {"$and": query_conditions} if len(query_conditions) > 1 else query_conditions[0]
        else:
            query = {}

        # Get transcripts
        transcripts = list(collection.find(query, {
            "meeting_date": 1,
            "UniqueName": 1,
            "Firm": 1,
            "transcripts": 1,
            "_id": 0
        }))

        return transcripts

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting transcripts: {e}")

# ====================================================================================
# SMART RAG STRATEGY
# ====================================================================================

# Global cache for RAG results (in-memory, per-session)
rag_cache = {}

# Global cache for all fund names (preloaded at startup)
ALL_FUND_NAMES = []
FUND_NAMES_LOADED = False

# Global BM25S indexes (pre-built at startup for instant search)
MEETING_BM25S_INDEX = None
MEETING_CHUNKS_CACHE = []  # All meeting chunks with metadata
MEETING_BM25S_LOADED = False

FACTSHEET_BM25S_INDEX = None
FACTSHEET_CHUNKS_CACHE = []  # All factsheet chunks with metadata
FACTSHEET_BM25S_LOADED = False

def load_all_fund_names():
    """
    Load all unique fund names from MeetingChunks into memory at startup.
    This eliminates the need to query the database on every fund name lookup.
    """
    global ALL_FUND_NAMES, FUND_NAMES_LOADED

    if FUND_NAMES_LOADED:
        return

    try:
        print("[FUND CACHE] Loading all fund names into memory...")
        db = mongo_client["Meetings"]
        chunks_collection = db["MeetingChunks"]

        # Get all unique fund names (no limit)
        pipeline = [
            {"$match": {"fund_name": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": "$fund_name"}},
        ]

        results = list(chunks_collection.aggregate(pipeline))
        ALL_FUND_NAMES = [r["_id"] for r in results if r["_id"] and isinstance(r["_id"], str)]

        FUND_NAMES_LOADED = True
        print(f"[FUND CACHE] Successfully loaded {len(ALL_FUND_NAMES)} unique fund names")
    except Exception as e:
        print(f"[FUND CACHE] Error loading fund names: {e}")
        ALL_FUND_NAMES = []


def load_meeting_bm25s_index():
    """
    Load all meeting chunks and build BM25S index at startup.
    This enables instant BM25 search without database queries.
    """
    global MEETING_BM25S_INDEX, MEETING_CHUNKS_CACHE, MEETING_BM25S_LOADED
    
    if MEETING_BM25S_LOADED:
        return
    
    try:
        import time
        start_time = time.time()
        print("[BM25S-MEETING] Loading all meeting chunks and building index...")
        
        db = mongo_client["Meetings"]
        chunks_collection = db["MeetingChunks"]
        
        # Fetch all meeting chunks with required fields
        chunks = list(chunks_collection.find(
            {},
            {
                "chunk_id": 1,
                "meeting_id": 1,
                "text": 1,
                "fund_name": 1,
                "manager": 1,
                "contact_person": 1,
                "meeting_date": 1,
                "strategy": 1,
                "importance": 1,
                "chunk_type": 1,
                "_id": 1
            }
        ))
        
        if not chunks:
            print("[BM25S-MEETING] No chunks found")
            MEETING_BM25S_LOADED = True
            return
        
        # Store chunks for later retrieval
        MEETING_CHUNKS_CACHE = chunks
        
        # Build corpus and index
        corpus = [chunk.get("text", "") for chunk in chunks]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        
        # Create BM25S retriever and index
        MEETING_BM25S_INDEX = bm25s.BM25()
        MEETING_BM25S_INDEX.index(corpus_tokens)
        
        load_time = time.time() - start_time
        MEETING_BM25S_LOADED = True
        print(f"[BM25S-MEETING] Successfully indexed {len(chunks)} chunks in {load_time:.2f}s")
        
    except Exception as e:
        print(f"[BM25S-MEETING] Error building index: {e}")
        import traceback
        traceback.print_exc()


def load_factsheet_bm25s_index():
    """
    Load all factsheet chunks and build BM25S index at startup.
    This enables instant BM25 search without database queries.
    """
    global FACTSHEET_BM25S_INDEX, FACTSHEET_CHUNKS_CACHE, FACTSHEET_BM25S_LOADED
    
    if FACTSHEET_BM25S_LOADED:
        return
    
    try:
        import time
        start_time = time.time()
        print("[BM25S-FACTSHEET] Loading all factsheet chunks and building index...")
        
        db = mongo_client["fund_reports"]
        chunks_collection = db["FactsheetChunks"]
        
        # Fetch all factsheet chunks with required fields (excluding full_document to save memory)
        chunks = list(chunks_collection.find(
            {},
            {
                "chunk_id": 1,
                "factsheet_id": 1,
                "text": 1,
                "UniqueName": 1,
                "fund_name": 1,
                "report_date": 1,
                "file_path": 1,
                "chunk_type": 1,
                "section_type": 1,
                "_id": 1
            }
        ))
        
        if not chunks:
            print("[BM25S-FACTSHEET] No chunks found")
            FACTSHEET_BM25S_LOADED = True
            return
        
        # Store chunks for later retrieval
        FACTSHEET_CHUNKS_CACHE = chunks
        
        # Build corpus and index
        corpus = [chunk.get("text", "") for chunk in chunks]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        
        # Create BM25S retriever and index
        FACTSHEET_BM25S_INDEX = bm25s.BM25()
        FACTSHEET_BM25S_INDEX.index(corpus_tokens)
        
        load_time = time.time() - start_time
        FACTSHEET_BM25S_LOADED = True
        print(f"[BM25S-FACTSHEET] Successfully indexed {len(chunks)} chunks in {load_time:.2f}s")
        
    except Exception as e:
        print(f"[BM25S-FACTSHEET] Error building index: {e}")
        import traceback
        traceback.print_exc()


def classify_intent(question: str, conversation_history: Optional[List[dict]] = None) -> str:
    """
    Classify user intent using rule-based patterns.

    Returns:
        'NO_RAG': Transform/analyze existing context
        'NEW_QUERY': Requires new vector search
    """
    question_lower = question.lower().strip()

    # NO_RAG patterns: Transform/analyze existing context
    transform_patterns = [
        'summarize', 'summary', 'list', 'compare', 'contrast',
        'tell me more', 'explain', 'elaborate', 'what do you mean',
        'can you clarify', 'in other words', 'why', 'how',
        'what about the', 'regarding the', 'about the above'
    ]

    # Check if question asks to transform existing context
    for pattern in transform_patterns:
        if pattern in question_lower:
            # If conversation history is short, we need new context
            if not conversation_history or len(conversation_history) < 2:
                return 'NEW_QUERY'
            return 'NO_RAG'

    # Default: assume new query that needs RAG
    return 'NEW_QUERY'


def detect_scope_change(
    question: str,
    start_date: Optional[str],
    end_date: Optional[str],
    selected_funds: Optional[List[str]],
    previous_scope: Optional[dict]
) -> bool:
    """
    Detect if the query scope has changed (time, entity, topic).
    Uses LLM-based intent detection to determine topic changes.

    Performance Optimization:
    - Returns immediately if filters changed (date/funds) without calling LLM
    - Only calls expensive Gemini API when filters are identical (200-800ms saved)

    Returns:
        True if scope changed, False if scope is the same
    """
    if previous_scope is None:
        return True  # No previous scope, treat as changed

    # Extract filter fields from previous scope for comparison
    previous_filters = {
        'start_date': previous_scope.get('start_date'),
        'end_date': previous_scope.get('end_date'),
        'funds': previous_scope.get('funds')
    }

    current_filters = {
        'start_date': start_date,
        'end_date': end_date,
        'funds': sorted(selected_funds) if selected_funds else None
    }

    # Check if filters changed - if yes, skip expensive LLM call and return immediately
    if current_filters != previous_filters:
        print(f"[SCOPE] Filters changed: {previous_filters} -> {current_filters}")
        print(f"[SCOPE OPTIMIZATION] Skipping LLM call - filters changed (200-800ms saved)")
        return True

    # Filters are the same, now check if topic changed using LLM-based classification
    previous_question = previous_scope.get('question', '')

    if not previous_question:
        return True  # No previous question to compare

    # Use Gemini to determine if the topic has changed
    try:
        classification_prompt = f"""You are a topic change detector for a conversational AI system.

Previous question: "{previous_question}"
Current question: "{question}"

Task: Determine if these two questions are about the SAME topic or DIFFERENT topics.

Guidelines:
- SAME topic: Follow-up questions, requests for more details, clarifications, or elaborations about the EXACT SAME subject/entity
  Examples: "Tell me about AQR" → "More details" (SAME)
           "What is their strategy?" → "Can you elaborate?" (SAME)
           "Tell me about copper" → "What's the outlook for copper?" (SAME)
- DIFFERENT topic: Questions about different entities, companies, funds, subjects, or when the entity name changes
  Examples: "Tell me about AQR" → "Tell me about FengHe" (DIFFERENT)
           "What is AQR's strategy?" → "Tell me about Citadel" (DIFFERENT)
           "Tell me about copper" → "Tell me about Cooper Square" (DIFFERENT - copper is a commodity, Cooper Square is a fund)
           "copper" → "Cooper Square" (DIFFERENT - these are different entities despite similar spelling)

IMPORTANT: If the entity name is different (even if similar sounding), classify as DIFFERENT.

Respond with ONLY one word: "SAME" or "DIFFERENT"
"""

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=classification_prompt
        )

        classification = response.text.strip().upper()

        if classification == "DIFFERENT":
            print(f"[SCOPE] Topic changed (LLM classification: DIFFERENT)")
            print(f"[SCOPE] Previous: '{previous_question[:60]}...'")
            print(f"[SCOPE] Current:  '{question[:60]}...'")
            return True
        else:
            print(f"[SCOPE] Same topic (LLM classification: {classification})")
            return False

    except Exception as e:
        print(f"[SCOPE] Error in LLM topic classification: {e}")
        # Fallback: assume topic changed if classification fails
        return True


def get_rag_strategy(
    question: str,
    start_date: Optional[str],
    end_date: Optional[str],
    selected_funds: Optional[List[str]],
    conversation_history: Optional[List[dict]] = None
) -> str:
    """
    Determine RAG strategy based on intent and scope.

    Returns:
        'NO_RAG': Skip vector search, reuse context
        'RAG_REUSE': Use cached RAG results
        'RAG_NEW_QUERY': Perform new vector search
    """
    # Step 1: Classify intent
    intent = classify_intent(question, conversation_history)

    if intent == 'NO_RAG':
        print(f"[RAG STRATEGY] NO_RAG - Transform/analyze intent detected")
        return 'NO_RAG'

    # Step 2: Check scope change
    cache_key = f"{start_date}_{end_date}_{sorted(selected_funds) if selected_funds else 'all'}"
    print(f"[RAG STRATEGY] Cache key: {cache_key}")
    print(f"[RAG STRATEGY] Available cache keys: {list(rag_cache.keys())}")
    previous_scope = rag_cache.get(cache_key, {}).get('scope')
    print(f"[RAG STRATEGY] Previous scope: {previous_scope}")

    scope_changed = detect_scope_change(
        question,
        start_date,
        end_date,
        selected_funds,
        previous_scope
    )

    if scope_changed:
        print(f"[RAG STRATEGY] RAG_NEW_QUERY - Scope changed or first query")
        return 'RAG_NEW_QUERY'
    else:
        print(f"[RAG STRATEGY] RAG_REUSE - Scope unchanged, reusing cached results")
        return 'RAG_REUSE'


def cache_rag_results(
    question: str,
    start_date: Optional[str],
    end_date: Optional[str],
    selected_funds: Optional[List[str]],
    rag_chunks: List[dict]
):
    """Cache RAG results for reuse."""
    cache_key = f"{start_date}_{end_date}_{sorted(selected_funds) if selected_funds else 'all'}"
    rag_cache[cache_key] = {
        'scope': {
            'question': question,  # Store question for LLM-based topic detection
            'start_date': start_date,
            'end_date': end_date,
            'funds': sorted(selected_funds) if selected_funds else None
        },
        'chunks': rag_chunks,
        'timestamp': datetime.now()
    }
    print(f"[RAG CACHE] Cached {len(rag_chunks)} chunks for key: {cache_key}")


def get_cached_rag_results(
    start_date: Optional[str],
    end_date: Optional[str],
    selected_funds: Optional[List[str]]
) -> Optional[List[dict]]:
    """Retrieve cached RAG results."""
    cache_key = f"{start_date}_{end_date}_{sorted(selected_funds) if selected_funds else 'all'}"
    cached = rag_cache.get(cache_key)

    if cached:
        print(f"[RAG CACHE] Retrieved {len(cached['chunks'])} cached chunks")
        return cached['chunks']

    return None

# ====================================================================================
# VECTOR SEARCH
# ====================================================================================

def find_matching_fund_names(query: str, db=None) -> List[str]:
    """
    Find fund names that match the query using smart phrase-based matching with adjective context detection.
    Returns list of matching fund_name values.
    Note: db parameter kept for backward compatibility but no longer used.

    Uses 3 strategies in order:
    1. Exact phrase match - Full fund name in query
    2. Two consecutive words - With adjective context detection to avoid false positives
    3. Single distinctive word - 5+ chars, not adjectives
    """
    try:
        # Ensure fund names are loaded
        if not FUND_NAMES_LOADED:
            load_all_fund_names()

        safe_query = query.encode('ascii', 'ignore').decode('ascii')
        print(f"[FUND MATCH] Query: {safe_query}".encode('ascii', 'ignore').decode('ascii'), flush=True)

        query_lower = query.lower()

        # Adjective context detection: phrases that indicate words are being used as adjectives
        adjective_indicators = [
            'top positions', 'top holdings', 'top performing', 'top 10', 'top five',
            'best positions', 'best holdings', 'best performing',
            'latest positions', 'latest holdings', 'latest factsheet',
            'largest positions', 'largest holdings',
            'current positions', 'current holdings'
        ]

        # Check if query has adjective context (e.g., "what are the top positions")
        has_adjective_context = any(indicator in query_lower for indicator in adjective_indicators)

        # Common adjectives that shouldn't match fund names when used in adjective context
        common_adjectives = {'top', 'best', 'latest', 'largest', 'current', 'main', 'primary'}

        matching_funds = []

        for fund_name in ALL_FUND_NAMES:
            fund_name_lower = fund_name.lower()

            # Get all words from fund name (no filtering yet)
            fund_words_all = fund_name_lower.split()

            # Strategy 1: Exact phrase match (most reliable)
            if fund_name_lower in query_lower:
                matching_funds.append(fund_name)
                safe_fund = fund_name.encode('ascii', 'ignore').decode('ascii')
                print(f"[FUND MATCH] Strategy 1 - Exact phrase: {safe_fund}".encode('ascii', 'ignore').decode('ascii'), flush=True)
                continue

            # Strategy 2: Two consecutive words with adjective filtering
            # Check consecutive word pairs from the fund name
            matched_in_strategy2 = False
            for i in range(len(fund_words_all) - 1):
                word1 = fund_words_all[i]
                word2 = fund_words_all[i+1]
                two_word_phrase = f"{word1} {word2}"

                if two_word_phrase in query_lower:
                    # Skip if first word is a common adjective AND query has adjective context
                    # This prevents "top positions" from matching "Top Ace Capital"
                    if word1 in common_adjectives and has_adjective_context:
                        continue

                    matching_funds.append(fund_name)
                    safe_fund = fund_name.encode('ascii', 'ignore').decode('ascii')
                    safe_phrase = two_word_phrase.encode('ascii', 'ignore').decode('ascii')
                    print(f"[FUND MATCH] Strategy 2 - Two words: {safe_fund} (phrase: '{safe_phrase}')".encode('ascii', 'ignore').decode('ascii'), flush=True)
                    matched_in_strategy2 = True
                    break

            if matched_in_strategy2:
                continue

            # Strategy 3: Single distinctive word (5+ chars, not adjectives)
            for word in fund_words_all:
                # Only match words that are:
                # - Long enough to be distinctive (5+ chars)
                # - Not common adjectives
                if len(word) >= 5 and word not in common_adjectives:
                    if word in query_lower:
                        matching_funds.append(fund_name)
                        safe_fund = fund_name.encode('ascii', 'ignore').decode('ascii')
                        safe_word = word.encode('ascii', 'ignore').decode('ascii')
                        print(f"[FUND MATCH] Strategy 3 - Single word: {safe_fund} (word: '{safe_word}')".encode('ascii', 'ignore').decode('ascii'), flush=True)
                        break

        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for fund in matching_funds:
            if fund not in seen:
                seen.add(fund)
                unique_matches.append(fund)

        print(f"[FUND MATCH] Found {len(unique_matches)} matching funds (from {len(ALL_FUND_NAMES)} total)".encode('ascii', 'ignore').decode('ascii'), flush=True)

        return unique_matches

    except Exception as e:
        error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
        print(f"[FUND MATCH] Error: {error_msg}".encode('ascii', 'ignore').decode('ascii'), flush=True)
        # Fallback to empty list on error
        return []

def perform_vector_search(
    question: str,
    data_sources: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    selected_funds: Optional[List[str]] = None,
    top_k: int = 50
) -> List[dict]:
    """
    Use RAG vector search to find relevant meeting chunks.
    Implements hybrid search: combines fund name matching with vector similarity.

    Args:
        question: User's question
        data_sources: List of data sources to search
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        selected_funds: List of fund names to filter
        top_k: Number of top results to return

    Returns:
        List of relevant chunk documents
    """
    try:
        import time

        total_start = time.time()

        # Connect to MongoDB
        db_start = time.time()
        db = mongo_client["Meetings"]
        chunks_collection = db["MeetingChunks"]
        meetings_collection = db["MeetingNotes"]
        db_time = time.time() - db_start
        print(f"[TIME] DB Connection: {db_time:.3f}s")

        # STEP 1: Generate embedding
        embedding_start = time.time()
        embedding_service = EmbeddingService(GEMINI_API_KEY)
        print(f"Generating embedding for query: {question[:100]}...")
        query_embedding = embedding_service.generate_embedding(
            question,
            task_type="RETRIEVAL_QUERY"
        )
        embedding_time = time.time() - embedding_start
        print(f"[TIME] Embedding Generation: {embedding_time:.3f}s")

        # STEP 2: Build filter for cosmosSearch (pre-filtering)
        filter_start = time.time()
        cosmos_filter = {}

        if start_date and end_date:
            # Dates are stored as strings in ISO format (YYYY-MM-DD), use directly
            cosmos_filter["meeting_date"] = {
                "$gte": start_date,
                "$lte": end_date
            }
            print(f"Filtering by date: {start_date} to {end_date}")

        if selected_funds and len(selected_funds) > 0:
            cosmos_filter["fund_name"] = {"$in": selected_funds}
            print(f"Filtering by funds: {selected_funds}")

        filter_time = time.time() - filter_start
        print(f"[TIME] Filter Building: {filter_time:.3f}s")

        # STEP 3: Build vector search pipeline with filter inside cosmosSearch
        pipeline_start = time.time()
        if cosmos_filter:
            print(f"Performing ENN filtered vector search (exact match on subset)...")
            pipeline = [
                {
                    "$search": {
                        "cosmosSearch": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "k": top_k,
                            "filter": cosmos_filter,  # Pre-filter before vector search
                            "exact": True  # Use ENN for filtered searches - 50% faster per MS docs
                        },
                        "returnStoredSource": True
                    }
                },
                {
                    "$project": {
                        "chunk_id": 1,
                        "meeting_id": 1,
                        "text": 1,
                        "fund_name": 1,
                        "manager": 1,
                        "contact_person": 1,
                        "meeting_date": 1,
                        "strategy": 1,
                        "importance": 1,
                        "chunk_type": 1,
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]
        else:
            # No filters, search all
            print(f"Performing vector search on all meetings...")
            pipeline = [
                {
                    "$search": {
                        "cosmosSearch": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "k": top_k
                        },
                        "returnStoredSource": True
                    }
                },
                {
                    "$project": {
                        "chunk_id": 1,
                        "meeting_id": 1,
                        "text": 1,
                        "fund_name": 1,
                        "manager": 1,
                        "contact_person": 1,
                        "meeting_date": 1,
                        "strategy": 1,
                        "importance": 1,
                        "chunk_type": 1,
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]

        pipeline_time = time.time() - pipeline_start
        print(f"[TIME] Pipeline Building: {pipeline_time:.3f}s")

        # STEP 4 & 5: Execute vector search and BM25 in PARALLEL
        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        def run_vector_search():
            return list(chunks_collection.aggregate(pipeline))

        def run_bm25_search():
            return perform_bm25_search_on_meeting_chunks(
                query=question,
                chunks_collection=chunks_collection,
                start_date=start_date,
                end_date=end_date,
                selected_funds=selected_funds,
                limit=top_k
            )

        parallel_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(run_vector_search)
            bm25_future = executor.submit(run_bm25_search)
            vector_results = vector_future.result()
            bm25_results = bm25_future.result()
        parallel_time = time.time() - parallel_start
        print(f"[TIME] Parallel Search (Vector + BM25): {parallel_time:.3f}s")
        print(f"[VECTOR] Found {len(vector_results)} results")
        print(f"[BM25] Found {len(bm25_results)} results")

        # STEP 6: Merge results using RRF (Reciprocal Rank Fusion)
        merge_start = time.time()
        if bm25_results:
            results = merge_vector_and_bm25_results(
                vector_results=vector_results,
                bm25_results=bm25_results,
                k=60,
                vector_weight=0.5,
                bm25_weight=0.5,
                top_k=top_k
            )
            print(f"[HYBRID-MEETING] Merged vector + BM25 results")
        else:
            # Fallback to vector-only if BM25 fails
            results = vector_results
            print(f"[HYBRID-MEETING] Using vector-only (BM25 returned no results)")
        merge_time = time.time() - merge_start
        print(f"[TIME] Merge: {merge_time:.3f}s")

        print(f"Returning {len(results)} chunks")

        total_time = time.time() - total_start
        print(f"[TIME] TOTAL TIME: {total_time:.3f}s")
        print(f"[TIME] Breakdown - DB: {db_time:.3f}s | Embedding: {embedding_time:.3f}s | Filter: {filter_time:.3f}s | Pipeline: {pipeline_time:.3f}s | Parallel(Vec+BM25): {parallel_time:.3f}s | Merge: {merge_time:.3f}s")

        return results

    except Exception as e:
        print(f"Error in vector search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing vector search: {e}")

def fetch_factsheets(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    selected_funds: Optional[List[str]] = None,
    limit: int = 20
) -> List[dict]:
    """
    Fetch factsheet documents using filters (no vector search).
    Returns factsheet documents with all fields.

    Args:
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        selected_funds: List of fund names to filter
        limit: Maximum number of factsheets to return

    Returns:
        List of factsheet documents with all fields
    """
    try:
        import time
        start_time = time.time()

        # Connect to MongoDB
        db = mongo_client["fund_reports"]
        factsheets_collection = db["factsheet"]

        # Build filter query - only use valid factsheets with UniqueName
        # Note: Records without UniqueName field cannot be included in statistics
        query = {
            "is_fund_factsheet": "y",
            "UniqueName": {"$exists": True, "$ne": None}
        }

        if start_date and end_date:
            # reportDate is stored as datetime
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            query["reportDate"] = {
                "$gte": start_dt,
                "$lte": end_dt
            }
            print(f"[FACTSHEET] Filtering by date: {start_date} to {end_date}")

        if selected_funds and len(selected_funds) > 0:
            query["UniqueName"] = {"$in": selected_funds}
            print(f"[FACTSHEET] Filtering by funds: {selected_funds}")

        # Fetch factsheets, sorted by reportDate descending (most recent first)
        # Note: Fetch more than limit initially to account for duplicates
        raw_results = list(factsheets_collection.find(query).sort("reportDate", -1).limit(limit * 3))

        # Deduplicate based on (UniqueName, reportDate) combination
        seen = set()
        unique_results = []

        for factsheet in raw_results:
            # Create a key from UniqueName and reportDate
            unique_key = (
                factsheet.get("UniqueName", ""),
                factsheet.get("reportDate")
            )

            if unique_key not in seen:
                seen.add(unique_key)
                unique_results.append(factsheet)

                # Stop once we have enough unique results
                if len(unique_results) >= limit:
                    break

        elapsed = time.time() - start_time
        print(f"[FACTSHEET] Found {len(raw_results)} total, {len(unique_results)} unique factsheets in {elapsed:.3f}s")

        return unique_results

    except Exception as e:
        print(f"Error fetching factsheets: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def perform_bm25_search_on_chunks(
    query: str,
    chunks_collection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    selected_funds: Optional[List[str]] = None,
    limit: int = 100
) -> List[dict]:
    """
    Perform BM25S search on factsheet chunks using pre-built index.
    
    Uses the pre-indexed BM25S retriever for instant search (~10ms vs ~5-10s before).
    Falls back to loading index if not loaded.
    
    Args:
        query: Search query
        chunks_collection: MongoDB collection (unused with BM25S)
        start_date: Start date filter
        end_date: End date filter
        selected_funds: List of fund names to filter
        limit: Maximum results to return
        
    Returns:
        List of matching chunk documents with bm25_score
    """
    import time
    from datetime import datetime as dt
    start_time = time.time()
    
    try:
        # Check if BM25S index is loaded
        if not FACTSHEET_BM25S_LOADED or FACTSHEET_BM25S_INDEX is None:
            print("[BM25S-FACTSHEET] Index not loaded, loading now...")
            load_factsheet_bm25s_index()
            
        if not FACTSHEET_BM25S_LOADED or FACTSHEET_BM25S_INDEX is None or not FACTSHEET_CHUNKS_CACHE:
            print("[BM25S-FACTSHEET] Index still not available, returning empty")
            return []
        
        # Parse date filters once
        start_dt = dt.fromisoformat(start_date) if start_date else None
        end_dt = dt.fromisoformat(end_date) if end_date else None
        
        # Tokenize query
        query_tokens = bm25s.tokenize(query, stopwords="en")
        
        # Retrieve top candidates (retrieve more than limit to filter)
        retrieve_k = min(limit * 5, len(FACTSHEET_CHUNKS_CACHE))  # Get 5x to allow filtering
        results, scores = FACTSHEET_BM25S_INDEX.retrieve(query_tokens, corpus=FACTSHEET_CHUNKS_CACHE, k=retrieve_k)
        
        search_time = time.time() - start_time
        
        # Filter by date/funds and map results
        scored_chunks = []
        for i, (chunk, score) in enumerate(zip(results[0], scores[0])):
            # Date filter
            if start_dt and end_dt:
                chunk_date = chunk.get("report_date")
                if chunk_date:
                    # Handle both datetime objects and strings
                    if isinstance(chunk_date, str):
                        try:
                            chunk_date = dt.fromisoformat(chunk_date)
                        except:
                            pass
                    if hasattr(chunk_date, 'replace'):  # is datetime
                        if chunk_date < start_dt or chunk_date > end_dt:
                            continue
            
            # Fund filter
            if selected_funds and len(selected_funds) > 0:
                chunk_fund = chunk.get("UniqueName")
                if chunk_fund not in selected_funds:
                    continue
            
            # Skip zero scores
            if score <= 0:
                continue
                
            result = chunk.copy()
            result["bm25_score"] = float(score)
            scored_chunks.append(result)
            
            if len(scored_chunks) >= limit:
                break
        
        total_time = time.time() - start_time
        print(f"[BM25S-FACTSHEET] Found {len(scored_chunks)} results in {total_time:.3f}s (search: {search_time:.3f}s)")
        
        return scored_chunks
        
    except Exception as e:
        print(f"[BM25S-FACTSHEET] Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def perform_bm25_search_on_meeting_chunks(
    query: str,
    chunks_collection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    selected_funds: Optional[List[str]] = None,
    limit: int = 100
) -> List[dict]:
    """
    Perform BM25S search on meeting note chunks using pre-built index.

    Uses the pre-indexed BM25S retriever for instant search (~10ms vs ~10s before).
    Falls back to legacy search if index not loaded.

    Args:
        query: Search query
        chunks_collection: MongoDB MeetingChunks collection (unused with BM25S)
        start_date: Start date filter (YYYY-MM-DD string)
        end_date: End date filter (YYYY-MM-DD string)
        selected_funds: List of fund names to filter
        limit: Maximum results to return

    Returns:
        List of matching chunk documents with bm25_score
    """
    import time
    start_time = time.time()

    try:
        # Check if BM25S index is loaded
        if not MEETING_BM25S_LOADED or MEETING_BM25S_INDEX is None:
            print("[BM25S-MEETING] Index not loaded, loading now...")
            load_meeting_bm25s_index()

        if not MEETING_BM25S_LOADED or MEETING_BM25S_INDEX is None or not MEETING_CHUNKS_CACHE:
            print("[BM25S-MEETING] Index still not available, returning empty")
            return []

        # Tokenize query
        query_tokens = bm25s.tokenize(query, stopwords="en")

        # Retrieve top candidates (retrieve more than limit to filter by date/funds)
        retrieve_k = min(limit * 10, len(MEETING_CHUNKS_CACHE))  # Get 10x to allow date/fund filtering
        results, scores = MEETING_BM25S_INDEX.retrieve(query_tokens, corpus=MEETING_CHUNKS_CACHE, k=retrieve_k)

        search_time = time.time() - start_time

        # Filter by date, funds, and map results
        scored_chunks = []
        for i, (chunk, score) in enumerate(zip(results[0], scores[0])):
            # Date filter
            if start_date and end_date:
                chunk_date = chunk.get("meeting_date", "")
                if chunk_date and (chunk_date < start_date or chunk_date > end_date):
                    continue

            # Fund filter
            if selected_funds and len(selected_funds) > 0:
                chunk_fund = chunk.get("fund_name")
                if chunk_fund not in selected_funds:
                    continue

            # Skip zero scores
            if score <= 0:
                continue

            result = chunk.copy()
            result["bm25_score"] = float(score)
            scored_chunks.append(result)

            if len(scored_chunks) >= limit:
                break

        total_time = time.time() - start_time
        print(f"[BM25S-MEETING] Found {len(scored_chunks)} results in {total_time:.3f}s (search: {search_time:.3f}s)")

        return scored_chunks

    except Exception as e:
        print(f"[BM25S-MEETING] Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def merge_vector_and_bm25_results(
    vector_results: List[dict],
    bm25_results: List[dict],
    k: int = 60,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k: int = 50
) -> List[dict]:
    """
    Merge vector and BM25 search results using Reciprocal Rank Fusion (RRF).

    RRF is a simple but effective method that doesn't require score normalization.
    Formula: RRF_score = sum(1 / (k + rank)) for each result list

    Args:
        vector_results: Results from vector search (with 'score' field)
        bm25_results: Results from BM25 search (with 'bm25_score' field)
        k: RRF constant (default 60, standard value from literature)
        vector_weight: Weight for vector results
        bm25_weight: Weight for BM25 results
        top_k: Number of results to return

    Returns:
        Merged and re-ranked results
    """
    rrf_scores = {}
    chunk_data = {}

    # Process vector results
    for rank, result in enumerate(vector_results):
        chunk_key = str(result.get("chunk_id") or result.get("_id"))
        rrf_score = vector_weight * (1.0 / (k + rank + 1))

        if chunk_key not in rrf_scores:
            rrf_scores[chunk_key] = 0
            chunk_data[chunk_key] = result.copy()

        rrf_scores[chunk_key] += rrf_score
        chunk_data[chunk_key]["vector_rank"] = rank + 1
        chunk_data[chunk_key]["vector_score"] = result.get("score", 0)

    # Process BM25 results
    for rank, result in enumerate(bm25_results):
        chunk_key = str(result.get("chunk_id") or result.get("_id"))
        rrf_score = bm25_weight * (1.0 / (k + rank + 1))

        if chunk_key not in rrf_scores:
            rrf_scores[chunk_key] = 0
            chunk_data[chunk_key] = result.copy()

        rrf_scores[chunk_key] += rrf_score
        chunk_data[chunk_key]["bm25_rank"] = rank + 1
        chunk_data[chunk_key]["bm25_score"] = result.get("bm25_score", 0)

    # Sort by combined RRF score
    sorted_chunks = sorted(
        [(key, rrf_scores[key]) for key in rrf_scores],
        key=lambda x: x[1],
        reverse=True
    )

    # Build final results
    merged_results = []
    for chunk_key, rrf_score in sorted_chunks[:top_k]:
        result = chunk_data[chunk_key]
        result["hybrid_score"] = rrf_score
        result["score"] = rrf_score  # Override for downstream compatibility
        merged_results.append(result)

    # Log fusion results
    print(f"[HYBRID] RRF merged {len(vector_results)} vector + {len(bm25_results)} BM25 -> {len(merged_results)} final")

    # Show top 5 results with details
    for i, r in enumerate(merged_results[:5]):
        fund = r.get("fund_name", "Unknown")[:20]
        v_rank = r.get("vector_rank", "-")
        b_rank = r.get("bm25_rank", "-")
        v_score = r.get("vector_score", 0)
        b_score = r.get("bm25_score", 0)
        text_preview = r.get("text", "")[:50].replace("\n", " ")
        print(f"[HYBRID] #{i+1}: {fund} | VecRank:{v_rank} BM25Rank:{b_rank} | VecScore:{v_score:.3f} BM25:{b_score:.2f} | RRF:{r['hybrid_score']:.4f}")
        print(f"         Text: {text_preview}...")

    return merged_results


def perform_factsheet_vector_search(
    question: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    selected_funds: Optional[List[str]] = None,
    top_k: int = 50
) -> List[dict]:
    """
    Use HYBRID search (Vector + BM25) to find relevant factsheet chunks.
    Combines semantic understanding (vector) with exact keyword matching (BM25).

    Args:
        question: User's question
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        selected_funds: List of fund names to filter
        top_k: Number of top results to return

    Returns:
        List of relevant chunk documents
    """
    try:
        import time
        import math

        total_start = time.time()

        # Connect to MongoDB
        db_start = time.time()
        db = mongo_client["fund_reports"]
        chunks_collection = db["FactsheetChunks"]
        db_time = time.time() - db_start
        print(f"[FACTSHEET TIME] DB Connection: {db_time:.3f}s")

        # STEP 1: Generate embedding
        embedding_start = time.time()
        embedding_service = EmbeddingService(GEMINI_API_KEY)
        print(f"[FACTSHEET] Generating embedding for query: {question[:100]}...")
        query_embedding = embedding_service.generate_embedding(
            question,
            task_type="RETRIEVAL_QUERY"
        )
        embedding_time = time.time() - embedding_start
        print(f"[FACTSHEET TIME] Embedding Generation: {embedding_time:.3f}s")

        # STEP 2: Build filter for cosmosSearch (pre-filtering)
        filter_start = time.time()
        cosmos_filter = {}

        if start_date and end_date:
            # reportDate is stored as datetime in factsheets
            from datetime import datetime as dt
            start_dt = dt.fromisoformat(start_date)
            end_dt = dt.fromisoformat(end_date)
            cosmos_filter["report_date"] = {
                "$gte": start_dt,
                "$lte": end_dt
            }
            print(f"[FACTSHEET] Filtering by date: {start_date} to {end_date}")

        if selected_funds and len(selected_funds) > 0:
            cosmos_filter["UniqueName"] = {"$in": selected_funds}
            print(f"[FACTSHEET] Filtering by funds: {selected_funds}")

        filter_time = time.time() - filter_start
        print(f"[FACTSHEET TIME] Filter Building: {filter_time:.3f}s")

        # STEP 3: Build vector search pipeline with filter inside cosmosSearch
        pipeline_start = time.time()
        if cosmos_filter:
            print(f"[FACTSHEET] Performing ENN filtered vector search (exact match on subset)...")
            pipeline = [
                {
                    "$search": {
                        "cosmosSearch": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "k": top_k,
                            "filter": cosmos_filter,  # Pre-filter before vector search
                            "exact": True  # Use ENN for filtered searches - 50% faster per MS docs
                        },
                        "returnStoredSource": True
                    }
                },
                {
                    "$project": {
                        "chunk_id": 1,
                        "factsheet_id": 1,
                        "text": 1,
                        "UniqueName": 1,
                        "fund_name": 1,
                        "report_date": 1,
                        "file_path": 1,
                        "chunk_type": 1,
                        "section_type": 1,  # Section type for context formatting
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]
        else:
            # No filters, search all
            print(f"[FACTSHEET] Performing vector search on all factsheets...")
            pipeline = [
                {
                    "$search": {
                        "cosmosSearch": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "k": top_k
                        },
                        "returnStoredSource": True
                    }
                },
                {
                    "$project": {
                        "chunk_id": 1,
                        "factsheet_id": 1,
                        "text": 1,
                        "UniqueName": 1,
                        "fund_name": 1,
                        "report_date": 1,
                        "file_path": 1,
                        "chunk_type": 1,
                        "section_type": 1,  # Section type for context formatting
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]

        pipeline_time = time.time() - pipeline_start
        print(f"[FACTSHEET TIME] Pipeline Building: {pipeline_time:.3f}s")

        # STEP 4: Execute vector search
        search_start = time.time()
        vector_results = list(chunks_collection.aggregate(pipeline))
        search_time = time.time() - search_start
        print(f"[FACTSHEET TIME] Vector Search Execution: {search_time:.3f}s")
        print(f"[FACTSHEET] Vector search found {len(vector_results)} results")

        # STEP 5: Execute BM25 search in parallel
        bm25_start = time.time()
        bm25_results = perform_bm25_search_on_chunks(
            query=question,
            chunks_collection=chunks_collection,
            start_date=start_date,
            end_date=end_date,
            selected_funds=selected_funds,
            limit=top_k * 2  # Fetch more for better BM25 coverage
        )
        bm25_time = time.time() - bm25_start
        print(f"[FACTSHEET TIME] BM25 Search: {bm25_time:.3f}s")

        # STEP 6: Merge results using Reciprocal Rank Fusion (RRF)
        merge_start = time.time()
        if bm25_results:
            # Hybrid search: merge vector + BM25 with RRF
            # Using 50/50 weight - balanced between semantic and keyword matching
            results = merge_vector_and_bm25_results(
                vector_results=vector_results,
                bm25_results=bm25_results,
                vector_weight=0.5,  # 50% weight for semantic search
                bm25_weight=0.5,    # 50% weight for keyword matching
                top_k=top_k
            )
            print(f"[FACTSHEET] HYBRID search enabled (Vector + BM25)")
        else:
            # Fallback to pure vector if BM25 failed
            results = vector_results[:top_k]
            print(f"[FACTSHEET] Pure vector search (BM25 unavailable)")
        merge_time = time.time() - merge_start
        print(f"[FACTSHEET TIME] Result Merging: {merge_time:.3f}s")

        print(f"[FACTSHEET] Returning {len(results)} chunks")

        total_time = time.time() - total_start
        print(f"[FACTSHEET TIME] TOTAL TIME: {total_time:.3f}s")
        print(f"[FACTSHEET TIME] Breakdown - DB: {db_time:.3f}s | Embedding: {embedding_time:.3f}s | Filter: {filter_time:.3f}s | Pipeline: {pipeline_time:.3f}s | VectorSearch: {search_time:.3f}s | BM25: {bm25_time:.3f}s | Merge: {merge_time:.3f}s")

        return results

    except Exception as e:
        print(f"[FACTSHEET] Error in hybrid search: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def format_factsheets_for_context(factsheets: List[dict]) -> str:
    """
    Format factsheet documents for inclusion in chat context.
    Preserves the entire JSON structure of each factsheet.

    Args:
        factsheets: List of factsheet documents

    Returns:
        Formatted string with complete factsheet JSON data
    """
    if not factsheets:
        return ""

    import json
    context = "=== FACTSHEET DATA ===\n\n"

    for idx, fs in enumerate(factsheets, 1):
        # Convert ObjectId and datetime to strings for JSON serialization
        factsheet_copy = {}
        for key, value in fs.items():
            if key == '_id':
                # Skip MongoDB _id field
                continue
            elif isinstance(value, datetime):
                # Convert datetime to ISO format string
                factsheet_copy[key] = value.isoformat()
            elif key == 'file_path':
                # Extract only the filename from the full path
                import os
                filename = os.path.basename(value) if value else 'Unknown File'
                factsheet_copy[key] = filename
            else:
                factsheet_copy[key] = value

        # Get filename and date for the label
        file_path = fs.get('file_path', '')
        if file_path:
            import os
            filename = os.path.basename(file_path)
        else:
            filename = 'Unknown File'

        report_date = fs.get('reportDate')
        if isinstance(report_date, datetime):
            report_date_str = report_date.strftime("%b %d, %Y")
        else:
            report_date_str = 'Unknown Date'

        # Convert to pretty-printed JSON with filename and date in label
        context += f"Factsheet - {filename} ({report_date_str}):\n"
        context += json.dumps(factsheet_copy, indent=2, ensure_ascii=False)
        context += "\n\n" + "="*80 + "\n\n"

    return context


def format_factsheet_chunks_for_context(chunks: List[dict]) -> str:
    """
    Format factsheet results from vector search using chunk text only.

    Strategy:
    - Group chunks by factsheet_id (deduplication)
    - Use only the chunk text (no full document fetch)
    - For each unique factsheet, show all matching sections with their text

    Args:
        chunks: List of chunk documents from vector search

    Returns:
        Formatted string with deduplicated factsheet data
    """
    if not chunks:
        return ""

    # Step 1: Deduplicate by factsheet_id and collect sections
    factsheet_map = {}  # factsheet_id -> {sections: [...], fund_name, report_date}

    for chunk in chunks:
        factsheet_id = chunk.get("factsheet_id", "unknown")
        section_type = chunk.get("section_type", "unknown")
        score = chunk.get("score", 0.0)
        chunk_text = chunk.get("text", "")

        if factsheet_id not in factsheet_map:
            factsheet_map[factsheet_id] = {
                "fund_name": chunk.get("fund_name") or chunk.get("UniqueName", "Unknown"),
                "report_date": chunk.get("report_date"),
                "file_path": chunk.get("file_path", ""),
                "sections": []
            }

        # Add this section with full text
        factsheet_map[factsheet_id]["sections"].append({
            "section_type": section_type,
            "score": score,
            "text": chunk_text  # Full text, not truncated
        })

    print(f"[FACTSHEET] Found {len(factsheet_map)} unique factsheets from {len(chunks)} chunks")

    # Step 2: Format results
    context = "=== RELEVANT FACTSHEET DATA (RAG Retrieved) ===\n\n"
    context += f"Found {len(factsheet_map)} unique factsheet(s) with {len(chunks)} relevant section(s)\n\n"

    for idx, (factsheet_id, data) in enumerate(factsheet_map.items(), 1):
        fund_name = data["fund_name"]
        sections = data["sections"]

        # Sort sections by score (highest first)
        sections.sort(key=lambda x: x["score"], reverse=True)

        # Get report date for label
        report_date = data["report_date"]
        if isinstance(report_date, datetime):
            report_date_str = report_date.strftime("%b %d, %Y")
        else:
            report_date_str = str(report_date) if report_date else "Unknown Date"

        # Get filename
        file_path = data.get("file_path", "")
        if file_path:
            import os
            filename = os.path.basename(file_path)
        else:
            filename = "Unknown File"

        # Header
        context += f"[{fund_name} Factsheet ({report_date_str})] - {filename}\n"
        context += f"Sections ({len(sections)}):\n\n"

        # Show each section's full text
        for sec in sections:
            context += f"--- {sec['section_type']} (relevance: {sec['score']:.4f}) ---\n"
            context += f"{sec['text']}\n\n"

        context += "="*80 + "\n\n"

    return context

def ask_gemini_with_rag(
    question: str,
    data_sources: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    selected_funds: Optional[List[str]] = None,
    conversation_history: Optional[List[dict]] = None
):
    """Send question with RAG-retrieved data to Gemini."""
    try:

        # Prepare context
        context = "Available Data Sources:\n\n"

        # Use RAG for meeting notes with smart strategy
        if "Meeting Notes" in data_sources:
            # Determine RAG strategy
            strategy = get_rag_strategy(
                question=question,
                start_date=start_date,
                end_date=end_date,
                selected_funds=selected_funds,
                conversation_history=conversation_history
            )

            if strategy == 'NO_RAG':
                # Skip RAG, reuse conversation context
                print("[RAG] Skipping vector search - using conversation context")
                rag_chunks = []
            elif strategy == 'RAG_REUSE':
                # Reuse cached RAG results
                rag_chunks = get_cached_rag_results(start_date, end_date, selected_funds)
                if not rag_chunks:
                    # Cache miss, perform new search
                    print("[RAG] Cache miss, performing new vector search...")
                    rag_chunks = perform_vector_search(
                        question=question,
                        data_sources=data_sources,
                        start_date=start_date,
                        end_date=end_date,
                        selected_funds=selected_funds,
                        top_k=50
                    )
                    cache_rag_results(question, start_date, end_date, selected_funds, rag_chunks)
            else:  # RAG_NEW_QUERY
                # Perform new vector search
                print("[RAG] Performing new vector search...")
                rag_chunks = perform_vector_search(
                    question=question,
                    data_sources=data_sources,
                    start_date=start_date,
                    end_date=end_date,
                    selected_funds=selected_funds,
                    top_k=50
                )
                cache_rag_results(question, start_date, end_date, selected_funds, rag_chunks)

            if rag_chunks:
                context += "=== MEETING NOTES (RAG Retrieved) ===\n\n"
                for chunk in rag_chunks:
                    meeting_date = chunk.get('meeting_date')
                    if isinstance(meeting_date, datetime):
                        meeting_date = meeting_date.strftime("%Y-%m-%d")

                    context += f"Date: {meeting_date}\n"
                    context += f"Fund: {chunk.get('fund_name', 'Unknown')}\n"
                    context += f"Manager: {chunk.get('manager', 'Unknown')}\n"
                    context += f"Chunk Type: {chunk.get('chunk_type', 'Unknown')}\n"
                    context += f"Content: {chunk.get('text', '')}\n"
                    context += f"Relevance Score: {chunk.get('score', 0):.4f}\n"
                    context += "\n---\n\n"

                print(f"Added {len(rag_chunks)} RAG chunks to context")
            else:
                print("No RAG chunks found")

        # TODO: Implement RAG for factsheets and transcripts later
        # For now, commenting out to test Meeting Notes RAG only

        # comment_data = []
        # transcript_data = []

        # # Fetch factsheet comments if needed
        # if "Factsheet Comments" in data_sources:
        #     comment_data = get_filtered_fund_comments(start_date, end_date, selected_funds)

        # # Add factsheet comments
        # if "Factsheet Comments" in data_sources and comment_data:
        #     context += "=== MONTHLY FACTSHEET COMMENTS ===\n\n"
        #     for comment in comment_data:
        #         context += f"Report Date: {comment.get('reportDate', 'Unknown')}\n"
        #         context += f"Fund: {comment.get('UniqueName', 'Unknown')} ({comment.get('fund_name', 'Unknown')})\n"
        #         context += f"Comment: {comment.get('comment', '')}\n"
        #         context += "\n---\n\n"

        # # Fetch transcripts if needed
        # if "Transcripts" in data_sources:
        #     transcript_data = get_filtered_transcripts(start_date, end_date, selected_funds)

        # # Add transcripts
        # if "Transcripts" in data_sources and transcript_data:
        #     context += "=== MEETING TRANSCRIPTS ===\n\n"
        #     for transcript in transcript_data:
        #         meeting_date = transcript.get('meeting_date', 'Unknown')
        #         if hasattr(meeting_date, 'strftime'):
        #             meeting_date = meeting_date.strftime("%Y-%m-%d")

        #         context += f"Date: {meeting_date}\n"
        #         context += f"Fund: {transcript.get('UniqueName', 'Unknown')}\n"
        #         context += f"Firm: {transcript.get('Firm', 'Unknown')}\n"
        #         context += f"Transcript: {transcript.get('transcripts', '')}\n"
        #         context += "\n---\n\n"

        # Build prompt with conversation history
        full_prompt = ""

        if conversation_history:
            full_prompt += "Previous conversation:\n\n"
            for item in conversation_history[-3:]:  # Last 3 exchanges
                full_prompt += f"User: {item.get('question', '')}\n"
                full_prompt += f"Assistant: {item.get('answer', '')}\n\n"
            full_prompt += "---\n\n"

        # Add current question with context
        full_prompt += f"""Based on the following data sources, please answer this question: {question}

Important instructions:
- PRIORITIZE internal information (meeting notes and factsheets) over web search results
- Always check the provided data sources first before using web search
- If internal information is available, present it first and prominently
- Use web search only to supplement or provide additional context when internal information is insufficient
- Use information from our conversation history if the question refers to previous discussion
- Cite specific sources when possible (meeting dates, report dates, fund names)
- Be concise and factual
- Distinguish between meeting notes and factsheets in your response

Available data sources: {', '.join(data_sources)}

{context}"""

        # Send to Gemini (non-streaming version for backward compatibility)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )

        return response.text.strip()

    except Exception as e:
        print(f"ERROR in ask_gemini: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {e}")

def ask_gemini_with_rag_streaming(
    question: str,
    data_sources: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    selected_funds: Optional[List[dict]] = None,
    conversation_history: Optional[List[dict]] = None,
    model: str = "gemini-2.5-flash"
):
    """Send question with RAG-retrieved data to Gemini with streaming support."""
    import time
    from google.genai import types
    overall_start = time.time()

    # Map frontend model names to actual API model names and thinking config
    actual_model = model
    thinking_config = None

    if model == "gemini-3-low-thinking":
        actual_model = "gemini-3-pro-preview"
        thinking_config = types.ThinkingConfig(thinking_level="low")
        print(f"[MODEL] Using {actual_model} with low thinking")
    elif model == "gemini-3-high-thinking":
        actual_model = "gemini-3-pro-preview"
        thinking_config = None
        print(f"[MODEL] Using {actual_model} with default (high) thinking")
    else:
        print(f"[MODEL] Using {actual_model}")

    try:
        print(f"\n{'='*80}")
        print(f"[TIMING] REQUEST START - Question: {question[:50]}...")
        print(f"[TIMING] Data sources: {data_sources}")
        print(f"{'='*80}\n")

        # Send searching status immediately so user sees feedback right away
        yield ('STATUS', True)

        print(f"[DEBUG] ask_gemini_with_rag_streaming called with data_sources: {data_sources}")

        # Prepare context (same as non-streaming version)
        context = "Available Data Sources:\n\n"

        # Track if we're performing vector search
        is_searching = False

        # Use RAG for meeting notes with smart strategy
        if "Meeting Notes" in data_sources:
            # Determine RAG strategy
            strategy = get_rag_strategy(
                question=question,
                start_date=start_date,
                end_date=end_date,
                selected_funds=selected_funds,
                conversation_history=conversation_history
            )

            if strategy == 'NO_RAG':
                # Skip RAG, reuse conversation context
                print("[RAG] Skipping vector search - using conversation context")
                rag_chunks = []
                # Send status: not searching
                yield ('STATUS', False)
            elif strategy == 'RAG_REUSE':
                # Reuse cached RAG results
                rag_chunks = get_cached_rag_results(start_date, end_date, selected_funds)
                if not rag_chunks:
                    # Cache miss, perform new search
                    print("[RAG] Cache miss, performing new vector search...")
                    # Send status: searching
                    yield ('STATUS', True)
                    rag_chunks = perform_vector_search(
                        question=question,
                        data_sources=data_sources,
                        start_date=start_date,
                        end_date=end_date,
                        selected_funds=selected_funds,
                        top_k=50
                    )
                    # Search complete, hide status
                    yield ('STATUS', False)
                    cache_rag_results(question, start_date, end_date, selected_funds, rag_chunks)
                else:
                    # Cache hit, no searching needed
                    yield ('STATUS', False)
            else:  # RAG_NEW_QUERY
                # Perform new vector search
                print("[RAG] Performing new vector search...")
                # Send status: searching
                yield ('STATUS', True)
                rag_chunks = perform_vector_search(
                    question=question,
                    data_sources=data_sources,
                    start_date=start_date,
                    end_date=end_date,
                    selected_funds=selected_funds,
                    top_k=50
                )
                # Search complete, hide status
                yield ('STATUS', False)
                cache_rag_results(question, start_date, end_date, selected_funds, rag_chunks)

            if rag_chunks:
                context += "=== MEETING NOTES (RAG Retrieved) ===\n\n"
                for chunk in rag_chunks:
                    meeting_date = chunk.get('meeting_date')
                    if isinstance(meeting_date, datetime):
                        meeting_date = meeting_date.strftime("%Y-%m-%d")

                    context += f"Date: {meeting_date}\n"
                    context += f"Fund: {chunk.get('fund_name', 'Unknown')}\n"
                    context += f"Manager: {chunk.get('manager', 'Unknown')}\n"
                    context += f"Chunk Type: {chunk.get('chunk_type', 'Unknown')}\n"
                    context += f"Content: {chunk.get('text', '')}\n"
                    context += f"Relevance Score: {chunk.get('score', 0):.4f}\n"
                    context += "\n---\n\n"

                print(f"Added {len(rag_chunks)} RAG chunks to context")
                # Debug: Print first chunk to verify content
                if rag_chunks:
                    print(f"DEBUG - First chunk text length: {len(rag_chunks[0].get('text', ''))}")
                    print(f"DEBUG - First chunk text preview: {rag_chunks[0].get('text', '')[:200]}")
            else:
                print("No RAG chunks found")
        else:
            # No Meeting Notes data source selected, no RAG
            yield ('STATUS', False)

        # Fetch factsheet data if selected (using RAG vector search)
        if "Factsheet" in data_sources:
            step_start = time.time()
            print("[FACTSHEET] Performing RAG vector search on factsheets...")
            factsheet_chunks = perform_factsheet_vector_search(
                question=question,
                start_date=start_date,
                end_date=end_date,
                selected_funds=selected_funds,
                top_k=50
            )
            factsheet_time = time.time() - step_start
            if factsheet_chunks:
                factsheet_context = format_factsheet_chunks_for_context(factsheet_chunks)
                context += factsheet_context
                print(f"[TIMING] Factsheet RAG: {factsheet_time:.3f}s ({len(factsheet_chunks)} chunks)")
            else:
                print(f"[TIMING] Factsheet RAG: {factsheet_time:.3f}s (no chunks found)")

        # Web Search will be handled by Google Search Grounding tool (configured below)
        use_google_search = "Web Search" in data_sources
        if use_google_search:
            print("[WEB SEARCH] Google Search Grounding enabled")

        # Debug: Print context length being sent to Gemini
        print(f"DEBUG - Total context length: {len(context)} characters")
        # Skip context preview to avoid encoding issues with special characters
        # print(f"DEBUG - Context preview (first 500 chars):\n{context[:500]}")

        # Build prompt with conversation history
        step_start = time.time()
        full_prompt = ""

        # Debug: Check conversation history (avoid printing content with unicode)
        print(f"[DEBUG] Conversation history received: {len(conversation_history) if conversation_history else 0} messages")
        if conversation_history:
            full_prompt += "Previous conversation:\n\n"
            for item in conversation_history[-3:]:  # Last 3 exchanges
                full_prompt += f"User: {item.get('question', '')}\n"
                full_prompt += f"Assistant: {item.get('answer', '')}\n\n"
            full_prompt += "---\n\n"

        # Add current question with context
        web_search_instructions = ""
        if use_google_search:
            web_search_instructions = """
3. **CRITICAL - Context-Aware Web Search (MANDATORY)**:
   - FIRST: Read and analyze ALL the internal data below to understand what entities (funds, companies, managers) are mentioned
   - THEN: Use web search to find additional current information about those SPECIFIC entities
   - **DO NOT** search based only on the user's question - use the internal data context
   - **Example**: If user asks "what about Cooper?" and internal data mentions "Cooper Square fund managed by Chad Clark at Select Equity Group":
     * GOOD searches: "Cooper Square investment fund", "Chad Clark Select Equity Group", "Cooper Square fund performance 2024"
     * BAD searches: "Cooper" (too generic, will return irrelevant results)
   - **CITATION REQUIREMENTS FOR INTERNAL DATA**:
     * CRITICAL: Copy the EXACT label from each data source - do not paraphrase or shorten
     * Factsheets are labeled as "Factsheet - [filename] ([date])" - use this EXACT format in citations
     * FORMATTING: Use italic markdown format for citations: *Source: Factsheet - AKO Global Fund.pdf (Oct 31, 2025)*
     * Example: If data is from "Factsheet - AKO Global Fund.pdf (Oct 31, 2025)", cite it as *Source: Factsheet - AKO Global Fund.pdf (Oct 31, 2025)*
     * Meeting notes include dates, fund names, and managers - include ALL these details in citations using the same italic format
     * Use inline citations throughout your analysis, not just at the end
     * Make citations specific and detailed so users know exactly where information came from
   - **STRUCTURE YOUR RESPONSE IN TWO CLEAR SECTIONS**:
     * **Section 1 - Internal Data Analysis**: Present all information from meeting notes and factsheets with detailed inline citations
     * **Section 2 - Web Search Results**: Present additional current information found via web search
       - MANDATORY: Every web search fact MUST include clickable markdown links to sources
       - Use markdown link format: [Source Title](https://full-url-here.com)
       - Example: "According to the latest filing from [AKO Capital 13F Q3 2025](https://sec.gov/cgi-bin/browse-edgar?action=getcompany...)"
       - Keep link text concise (under 50 characters) but descriptive
       - DO NOT write web search analysis without citing the specific webpage links
       - If you cannot find sources to cite, state "No relevant web sources found" instead of writing unsourced content
       - **CRITICAL - NUMERICAL DATA VERIFICATION**: Financial numbers and percentages are HIGHLY sensitive and must be directly verifiable
         * When citing ANY number (stock prices, percentages, returns, AUM, etc.), the user MUST be able to see that EXACT number on the linked page
         * Example: If you state "AppLovin rose 64%", when the user clicks the link, they must see "64%" or the specific data that supports this claim
         * DO NOT cite numbers that are not directly visible or quoted in the source page
         * If you cannot verify a specific number in the source, DO NOT include it - use qualitative language instead ("rose significantly") or omit the claim
         * Add the exact quote from the source page in parentheses after the link if the number is buried in text
         * Example: "AppLovin stock increased [source link] (page states: 'APP gained 64% in October 2025')"
   - Web searches MUST be relevant to entities found in internal data
"""
        else:
            web_search_instructions = """
3. Only use the internal data provided below. Web search is not enabled.
"""

        full_prompt += f"""Question: {question}

Instructions:
1. **FORMATTING - CRITICAL**:
   - Use ONLY standard markdown formatting - NO HTML tags
   - For line breaks within lists, use double newlines between items
   - For tables, use proper markdown table syntax with pipes (|)
   - NEVER use HTML tags like <br>, <table>, <div>, etc.
2. **DATE AWARENESS - CRITICAL**:
   - Financial data is highly time-sensitive - always check the dates of data sources
   - When answering questions about "current" or "latest" positions, performance, or holdings, prioritize the MOST RECENT factsheets and meeting notes
   - Clearly state the date of the data you're referencing (e.g., "As of October 2025..." or "Based on the October 31, 2025 factsheet...")
   - If comparing data across time periods, explicitly mention the dates being compared
   - Be aware that factsheets may be dated differently - use the most recent available data
3. Read and analyze the internal data sources provided below (meeting notes and factsheets)
4. Present relevant internal information with comprehensive details
{web_search_instructions}
5. Synthesize all information (internal + web if enabled) into one comprehensive answer
6. Provide thorough explanations with context, background, and analysis
7. Cite all sources with specifics (meeting dates, URLs, fund names, manager names)
8. Use clear section headers to distinguish internal data from web search results
9. Give detailed, comprehensive responses - do not be brief
10. **CITATION STYLE - IMPORTANT**: When citing factsheets, ALWAYS use the fund name and date (e.g., "Infinitum Master Fund Factsheet (Sep 01, 2025)"). NEVER use generic numbered references like "Factsheet 1" or "(Source: Factsheet 16)" - these are meaningless to users

Available data sources: {', '.join(data_sources)}

Internal Data:
{context}

Now provide a comprehensive, detailed answer to the question above."""

        # Debug: Print the final prompt being sent to Gemini
        prompt_build_time = time.time() - step_start
        print(f"[TIMING] Prompt building: {prompt_build_time:.3f}s")
        print(f"DEBUG - Final prompt length: {len(full_prompt)} characters")
        # Handle encoding for Windows console (replace non-ASCII chars)
        safe_preview = full_prompt[:1000].encode('ascii', errors='replace').decode('ascii')
        print(f"DEBUG - Final prompt preview (first 1000 chars):\n{safe_preview}")

        # Configure Google Search Grounding if web search is enabled
        step_start = time.time()
        config = None

        if use_google_search:
            # Use Google Search Grounding - Gemini automatically searches, reads, and synthesizes web content
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            if thinking_config:
                config = types.GenerateContentConfig(
                    tools=[google_search_tool],
                    temperature=1,  # Maximum creativity and varied responses
                    thinking_config=thinking_config
                )
            else:
                config = types.GenerateContentConfig(
                    tools=[google_search_tool],
                    temperature=1
                )
            print(f"[WEB SEARCH] Configured Google Search Grounding")
        else:
            # Configure for comprehensive responses with no length restrictions
            if thinking_config:
                config = types.GenerateContentConfig(
                    temperature=1,
                    thinking_config=thinking_config
                )
            else:
                config = types.GenerateContentConfig(
                    temperature=1
                )

        # Send status: generating response
        yield ('GENERATING', True)

        # Send to Gemini with streaming
        api_call_time = time.time() - step_start
        print(f"[TIMING] API config setup: {api_call_time:.3f}s")

        try:
            step_start = time.time()
            if config:
                response = gemini_client.models.generate_content_stream(
                    model=actual_model,
                    contents=full_prompt,
                    config=config
                )
            else:
                response = gemini_client.models.generate_content_stream(
                    model=actual_model,
                    contents=full_prompt
                )

            api_init_time = time.time() - step_start
            print(f"[TIMING] API call initiation: {api_init_time:.3f}s")

            first_chunk = True
            chunk_start = time.time()
            last_heartbeat_time = time.time()
            heartbeat_interval = 15  # Send heartbeat every 15 seconds during streaming
            for chunk in response:
                # Send heartbeat if it's been too long since last activity
                current_time = time.time()
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    yield ('HEARTBEAT', True)
                    last_heartbeat_time = current_time
                    print(f"[HEARTBEAT] Sent during Gemini streaming at {current_time:.0f}")

                if first_chunk:
                    first_chunk_time = time.time() - chunk_start
                    print(f"[TIMING] Time to first chunk: {first_chunk_time:.3f}s")
                    first_chunk = False
                    last_heartbeat_time = time.time()  # Reset after first chunk
                try:
                    # Debug: Print chunk structure
                    print(f"[DEBUG] Chunk type: {type(chunk)}")
                    print(f"[DEBUG] Chunk attributes: {dir(chunk)}")

                    # Try to access text directly first (for backwards compatibility)
                    if hasattr(chunk, 'text') and chunk.text:
                        print(f"[STREAMING] Yielding chunk: {len(chunk.text)} chars")
                        yield chunk.text
                    # Try to extract from candidates.content.parts
                    elif hasattr(chunk, 'candidates'):
                        for candidate in chunk.candidates:
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    print(f"[DEBUG] Part type: {type(part)}, attributes: {dir(part)}")

                                    # Check if this is a thought summary
                                    if hasattr(part, 'thought') and part.thought:
                                        # Yield thought summary with special marker
                                        if hasattr(part, 'text') and part.text:
                                            print(f"[STREAMING] Yielding thought summary: {len(part.text)} chars")
                                            yield f"__THOUGHT__{part.text}__END_THOUGHT__"
                                    # Regular content
                                    elif hasattr(part, 'text') and part.text:
                                        print(f"[STREAMING] Extracted text from parts: {len(part.text)} chars")
                                        yield part.text
                    else:
                        print(f"[STREAMING] Chunk has no accessible text")
                except Exception as chunk_error:
                    print(f"[STREAMING ERROR] Error processing chunk: {chunk_error}")
                    import traceback
                    traceback.print_exc()

        except Exception as api_error:
            # Handle Gemini API errors (rate limits, overload, etc.)
            error_msg = str(api_error)
            print(f"[API ERROR] Gemini API error: {error_msg}")

            if "503" in error_msg or "overloaded" in error_msg.lower():
                yield "⚠️ The Gemini API is currently overloaded. Please try again in a few moments, or disable web search to use only internal data."
            elif "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                yield "⚠️ API rate limit reached. Please wait a moment before trying again, or disable web search to use only internal data."
            else:
                yield f"⚠️ API Error: {error_msg}"

    except Exception as e:
        print(f"ERROR in ask_gemini_streaming: {str(e)}")
        import traceback
        traceback.print_exc()
        yield f"⚠️ Unexpected error: {str(e)}"
    finally:
        total_time = time.time() - overall_start
        print(f"\n{'='*80}")
        print(f"[TIMING] TOTAL REQUEST TIME: {total_time:.3f}s")
        print(f"{'='*80}\n")

# Portfolio helper functions
STERWEN_BEST_IDEAS_FUNDS = [
    "Infinitum Partners", "Golden Pine", "Camber Capital", "Callodine Capital",
    "Hiddenite Capital", "FengHe", "Arini", "Varecs Partners", "Astaris Capital",
    "EDL Capital", "Pertento Partners", "Gemsstock Fund", "Whitebox Advisors",
    "Contour Manticore", "Teleios Global Opp.", "CFM", "Crake Asset Management",
    "Permian", "AKO Global", "Kings Court Capital", "Castle Hook"
]

def get_all_portfolios():
    """Get list of all unique portfolios."""
    try:
        all_portfolios = set()

        # Get LODH portfolios
        lodh_db = mongo_client["LODH"]
        lodh_collection = lodh_db["Positions"]
        lodh_latest = lodh_collection.find_one(sort=[("valuationDate", -1)])
        if lodh_latest and "valuationDate" in lodh_latest:
            lodh_date = lodh_latest["valuationDate"]
            lodh_records = list(lodh_collection.find({"valuationDate": lodh_date}))
            for record in lodh_records:
                portfolio = record.get("portfolio")
                if portfolio:
                    portfolio_str = str(portfolio).strip()
                    if portfolio_str not in ["nan", "None", "Unknown", ""]:
                        all_portfolios.add(portfolio_str)

        # Get Pictet portfolios
        pictet_db = mongo_client["Pictet"]
        pictet_collection = pictet_db["Positions"]
        pictet_latest = pictet_collection.find_one(sort=[("valuation_date", -1)])
        if pictet_latest and "valuation_date" in pictet_latest:
            pictet_date = pictet_latest["valuation_date"]
            pictet_records = list(pictet_collection.find({"valuation_date": pictet_date}))
            for record in pictet_records:
                portfolio = record.get("portfolio")
                if portfolio:
                    portfolio_str = str(portfolio).strip()
                    if portfolio_str not in ["nan", "None", "Unknown", ""]:
                        all_portfolios.add(portfolio_str)

        sorted_portfolios = sorted(list(all_portfolios))
        return ["SterWen Best Ideas"] + sorted_portfolios

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting portfolios: {e}")

def get_funds_by_portfolios(selected_portfolios: List[str]):
    """Get all unique funds belonging to selected portfolios."""
    if not selected_portfolios:
        return []

    try:
        all_funds = set()

        # Handle custom "SterWen Best Ideas" portfolio
        if "SterWen Best Ideas" in selected_portfolios:
            all_funds.update(STERWEN_BEST_IDEAS_FUNDS)

        # Handle regular portfolios
        regular_portfolios = [p for p in selected_portfolios if p != "SterWen Best Ideas"]

        if regular_portfolios:
            # Get LODH funds
            lodh_db = mongo_client["LODH"]
            lodh_collection = lodh_db["Positions"]
            lodh_latest = lodh_collection.find_one(sort=[("valuationDate", -1)])
            if lodh_latest and "valuationDate" in lodh_latest:
                lodh_date = lodh_latest["valuationDate"]
                lodh_records = list(lodh_collection.find({"valuationDate": lodh_date, "portfolio": {"$in": regular_portfolios}}))
                for record in lodh_records:
                    unique_name = record.get("UniqueName")
                    if unique_name:
                        fund_str = str(unique_name).strip()
                        if fund_str not in ["nan", "None", "Unknown", ""]:
                            all_funds.add(fund_str)

            # Get Pictet funds
            pictet_db = mongo_client["Pictet"]
            pictet_collection = pictet_db["Positions"]
            pictet_latest = pictet_collection.find_one(sort=[("valuation_date", -1)])
            if pictet_latest and "valuation_date" in pictet_latest:
                pictet_date = pictet_latest["valuation_date"]
                pictet_records = list(pictet_collection.find({"valuation_date": pictet_date, "portfolio": {"$in": regular_portfolios}}))
                for record in pictet_records:
                    unique_name = record.get("UniqueName")
                    if unique_name:
                        fund_str = str(unique_name).strip()
                        if fund_str not in ["nan", "None", "Unknown", ""]:
                            all_funds.add(fund_str)

        return sorted(list(all_funds))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting funds by portfolios: {e}")

# API Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "meeting-notes-chatbot"}

@app.get("/api/chat/portfolios")
async def get_portfolios_endpoint():
    """Get list of all portfolios."""
    try:
        portfolios = get_all_portfolios()
        return {"portfolios": portfolios}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting portfolios: {e}")

@app.post("/api/chat/portfolios/funds")
async def get_funds_by_portfolios_endpoint(data: dict):
    """Get funds belonging to selected portfolios."""
    try:
        selected_portfolios = data.get("portfolios", [])
        funds = get_funds_by_portfolios(selected_portfolios)
        return {"funds": funds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting funds by portfolios: {e}")

@app.get("/api/chat/funds")
async def get_all_funds():
    """Get list of all unique funds."""
    try:
        # Get funds from meeting notes
        meetings_db = mongo_client["Meetings"]
        meetings_collection = meetings_db["Meeting Notes"]
        meeting_funds = meetings_collection.distinct("UniqueName")

        # Get funds from fund reports
        reports_db = mongo_client["fund_reports"]
        reports_collection = reports_db["factsheet"]
        report_funds = reports_collection.distinct("UniqueName", {"is_fund_factsheet": "y"})

        # Get funds from transcripts
        transcripts_collection = meetings_db["transcripts"]
        transcript_funds = transcripts_collection.distinct("UniqueName")

        # Combine and clean
        all_funds = set()
        for fund_list in [meeting_funds, report_funds, transcript_funds]:
            for f in fund_list:
                if f is not None:
                    fund_str = str(f).strip()
                    if fund_str and fund_str not in ["nan", "None", "Unknown", ""]:
                        all_funds.add(fund_str)

        return {"funds": sorted(list(all_funds))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting funds: {e}")

@app.post("/api/chat/stats", response_model=FilterStats)
async def get_filter_stats(request: ChatRequest):
    """Get counts based on current filters."""
    try:
        print(f"\n[STATS] Request filters:")
        print(f"  - Data sources: {request.data_sources}")
        print(f"  - Date range: {request.start_date} to {request.end_date}")
        print(f"  - Selected funds: {request.selected_funds}")

        meetings_count = 0
        factsheets_count = 0
        transcripts_count = 0

        if "Meeting Notes" in request.data_sources:
            meetings = get_filtered_meetings(request.start_date, request.end_date, request.selected_funds)
            meetings_count = len(meetings)

        if "Factsheet" in request.data_sources:
            # Use fetch_factsheets to get the actual count of complete factsheet documents
            # Set limit to a high number to get all matching factsheets for counting
            factsheets = fetch_factsheets(
                start_date=request.start_date,
                end_date=request.end_date,
                selected_funds=request.selected_funds,
                limit=1000  # High limit to get all factsheets for accurate count
            )
            factsheets_count = len(factsheets)
            print(f"[STATS] Factsheet count: {factsheets_count}")

        if "Transcripts" in request.data_sources:
            transcripts = get_filtered_transcripts(request.start_date, request.end_date, request.selected_funds)
            transcripts_count = len(transcripts)

        return FilterStats(
            meeting_notes_count=meetings_count,
            factsheet_comments_count=factsheets_count,
            transcripts_count=transcripts_count,
            total_count=meetings_count + factsheets_count + transcripts_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {e}")

@app.post("/api/chat/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """Ask a question with RAG-powered filtered data (non-streaming)."""
    try:
        # Use the new RAG-powered function
        print(f"\n=== New Question ===")
        print(f"Question: {request.question}")
        print(f"Data sources: {request.data_sources}")
        print(f"Date range: {request.start_date} to {request.end_date}")
        print(f"Selected funds: {request.selected_funds}")

        answer = ask_gemini_with_rag(
            question=request.question,
            data_sources=request.data_sources,
            start_date=request.start_date,
            end_date=request.end_date,
            selected_funds=request.selected_funds,
            conversation_history=request.conversation_history
        )

        # For data summary, we can't easily count the exact items without refetching
        # But we can provide a meaningful response
        return ChatResponse(
            answer=answer,
            sources_used=request.data_sources,
            data_summary={
                "meeting_notes": "RAG-powered" if "Meeting Notes" in request.data_sources else 0,
                "factsheet_comments": "Filtered" if "Factsheet" in request.data_sources else 0,
                "transcripts": "Filtered" if "Transcripts" in request.data_sources else 0
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {e}")

@app.post("/api/chat/ask/stream")
async def ask_question_stream(request: ChatRequest):
    """Ask a question with RAG-powered filtered data (streaming)."""
    try:
        print(f"\n=== New Streaming Question ===")
        print(f"Question: {request.question}")
        print(f"Data sources: {request.data_sources}")
        print(f"Date range: {request.start_date} to {request.end_date}")
        print(f"Selected funds: {request.selected_funds}")
        print(f"Model: {request.model}")

        def generate():
            import time
            try:
                print("[DEBUG] ========== GENERATE FUNCTION STARTED ==========")
                print(f"[DEBUG] Question: {request.question}")
                print(f"[DEBUG] Data sources: {request.data_sources}")
                print("[DEBUG] About to call ask_gemini_with_rag_streaming...")

                # Track last heartbeat time for Azure keep-alive (every 15 seconds)
                last_heartbeat = time.time()
                heartbeat_interval = 15  # Send heartbeat every 15 seconds

                for item in ask_gemini_with_rag_streaming(
                    question=request.question,
                    data_sources=request.data_sources,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    selected_funds=request.selected_funds,
                    conversation_history=request.conversation_history,
                    model=request.model
                ):
                    # Check if we need to send a heartbeat to keep Azure connection alive
                    current_time = time.time()
                    if current_time - last_heartbeat >= heartbeat_interval:
                        yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                        last_heartbeat = current_time
                        print(f"[HEARTBEAT] Sent heartbeat at {current_time:.0f}")

                    print(f"[DEBUG] Received item from streaming: {type(item)}")
                    # Check if it's a status tuple or text content
                    if isinstance(item, tuple) and item[0] == 'STATUS':
                        # Send search status event
                        yield f"data: {json.dumps({'searching': item[1]})}\n\n"
                        last_heartbeat = time.time()  # Reset heartbeat timer
                    elif isinstance(item, tuple) and item[0] == 'GENERATING':
                        # Send generating status event
                        yield f"data: {json.dumps({'generating': item[1]})}\n\n"
                        last_heartbeat = time.time()  # Reset heartbeat timer
                    elif isinstance(item, tuple) and item[0] == 'HEARTBEAT':
                        # Send heartbeat event from streaming function
                        yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                        last_heartbeat = time.time()
                        print(f"[HEARTBEAT] Sent heartbeat from streaming function")
                    elif isinstance(item, str) and '__THOUGHT__' in item:
                        # Extract thought content and send as thinking event
                        thought_content = item.replace('__THOUGHT__', '').replace('__END_THOUGHT__', '')
                        yield f"data: {json.dumps({'thinking': thought_content})}\n\n"
                        last_heartbeat = time.time()  # Reset heartbeat timer
                    else:
                        # Strip HTML tags and send each text chunk as Server-Sent Events format
                        cleaned_content = strip_html_tags(item) if isinstance(item, str) else item
                        yield f"data: {json.dumps({'content': cleaned_content})}\n\n"
                        last_heartbeat = time.time()  # Reset heartbeat timer

                # Send done signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                print("[DEBUG] Finished generate() function")
            except Exception as e:
                print(f"Error in generate: {str(e)}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "X-Content-Type-Options": "nosniff",  # Prevent MIME sniffing
                "Transfer-Encoding": "chunked",  # Ensure chunked transfer
                "Content-Encoding": "identity"  # Disable compression that could buffer
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing streaming question: {e}")

@app.post("/api/chat/conversations/create")
async def create_conversation(request: Request, create_request: CreateConversationRequest):
    """Create a new conversation."""
    try:
        user_id = get_user_identifier(request, create_request.session_id)
        conversation_id = str(uuid.uuid4())

        db = mongo_client["Chatbot"]
        collection = db["Conversations"]

        conversation_data = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": create_request.title,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        collection.insert_one(conversation_data)

        return {
            "success": True,
            "conversation_id": conversation_id,
            "title": create_request.title
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {e}")

@app.get("/api/chat/conversations")
async def list_conversations(request: Request, session_id: Optional[str] = None):
    """List all conversations for the current user."""
    try:
        user_id = get_user_identifier(request, session_id)

        db = mongo_client["Chatbot"]
        collection = db["Conversations"]

        # Find all conversations for this user, sorted by updated_at
        conversations = list(collection.find(
            {"user_id": user_id},
            {"_id": 0, "conversation_id": 1, "title": 1, "created_at": 1, "updated_at": 1, "messages": 1}
        ).sort("updated_at", -1))

        # Filter out empty conversations and convert datetime objects to strings
        non_empty_conversations = []
        for conv in conversations:
            # Skip conversations with no messages
            if len(conv.get("messages", [])) == 0:
                continue

            conv["created_at"] = conv["created_at"].isoformat() if "created_at" in conv else None
            conv["updated_at"] = conv["updated_at"].isoformat() if "updated_at" in conv else None
            conv["message_count"] = len(conv.get("messages", []))
            # Remove messages from list view for performance
            conv.pop("messages", None)
            non_empty_conversations.append(conv)

        return {
            "success": True,
            "conversations": non_empty_conversations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {e}")

@app.get("/api/chat/conversations/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str, session_id: Optional[str] = None):
    """Get a specific conversation."""
    try:
        user_id = get_user_identifier(request, session_id)

        db = mongo_client["Chatbot"]
        collection = db["Conversations"]

        conversation = collection.find_one(
            {"conversation_id": conversation_id, "user_id": user_id},
            {"_id": 0}
        )

        if conversation:
            # Convert datetime objects to strings
            conversation["created_at"] = conversation["created_at"].isoformat() if "created_at" in conversation else None
            conversation["updated_at"] = conversation["updated_at"].isoformat() if "updated_at" in conversation else None

            return {
                "success": True,
                "conversation": conversation
            }
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {e}")

@app.post("/api/chat/conversations/{conversation_id}/messages")
async def save_message(request: Request, conversation_id: str, chat_request: SaveChatRequest):
    """Save a message to a conversation."""
    try:
        user_id = get_user_identifier(request, chat_request.session_id)

        db = mongo_client["Chatbot"]
        collection = db["Conversations"]

        message_data = {
            "question": chat_request.message.question,
            "answer": chat_request.message.answer,
            "sources": chat_request.message.sources,
            "timestamp": chat_request.message.timestamp
        }

        # Update the conversation, set title from first question if still "New Conversation"
        update_doc = {
            "$push": {"messages": message_data},
            "$set": {"updated_at": datetime.utcnow()}
        }

        # Get current conversation to check title
        conversation = collection.find_one({"conversation_id": conversation_id, "user_id": user_id})
        if conversation and conversation.get("title") == "New Conversation" and len(conversation.get("messages", [])) == 0:
            # Generate title from first question
            title = chat_request.message.question[:50] + "..." if len(chat_request.message.question) > 50 else chat_request.message.question
            update_doc["$set"]["title"] = title

        result = collection.update_one(
            {"conversation_id": conversation_id, "user_id": user_id},
            update_doc
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "success": True,
            "message": "Message saved successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving message: {e}")

@app.delete("/api/chat/conversations/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str, session_id: Optional[str] = None):
    """Delete a conversation."""
    try:
        user_id = get_user_identifier(request, session_id)

        db = mongo_client["Chatbot"]
        collection = db["Conversations"]

        result = collection.delete_one({"conversation_id": conversation_id, "user_id": user_id})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "success": True,
            "message": "Conversation deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


