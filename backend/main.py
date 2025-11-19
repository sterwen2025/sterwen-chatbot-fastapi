"""
Meeting Notes & Fund Comments Chatbot API
==========================================
FastAPI backend for intelligent Q&A system with meeting notes, factsheets, and transcripts
"""

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date
from pymongo import MongoClient
import calendar
import os
import json
import uuid
from google import genai
from googleapiclient.discovery import build
from dotenv import load_dotenv
from embedding_service import EmbeddingService
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://mongodbsterwen:Genevaboy$1204@sterwendb.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

# Configure Gemini client (automatically uses GEMINI_API_KEY from environment)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Debug: Check if API key is loaded (don't print the key itself)
print(f"GEMINI_API_KEY loaded: {bool(GEMINI_API_KEY)}")
print("=" * 80)
print("CONTAINER VERSION: 2024-11-13-gemini-2.5-flash-switch")
print("=" * 80)

app = FastAPI(title="Meeting Notes Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    question: str
    data_sources: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    selected_funds: Optional[List[str]] = None
    conversation_history: Optional[List[dict]] = None

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

def normalize_date_to_month_end(date_input):
    """Convert any date to the last day of its month, except for current month."""
    if not date_input or date_input in ["", "None", None]:
        return None

    try:
        parsed_date = None

        # Handle datetime objects directly
        if isinstance(date_input, datetime):
            parsed_date = date_input
        elif isinstance(date_input, date):
            parsed_date = datetime.combine(date_input, datetime.min.time())
        else:
            # Try different date formats for strings
            date_formats = [
                "%Y-%m-%d",
                "%Y%m%d",
                "%Y/%m/%d",
                "%m/%d/%Y",
                "%d/%m/%Y"
            ]

            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(str(date_input), fmt)
                    break
                except ValueError:
                    continue

        if parsed_date:
            # Get current date
            current_date = datetime.now()

            # Check if the date is in the current month and year
            if (parsed_date.year == current_date.year and
                parsed_date.month == current_date.month):
                # Keep the original date for current month
                return parsed_date.strftime("%Y-%m-%d")
            else:
                # Normalize to last day of month for past months
                last_day = calendar.monthrange(parsed_date.year, parsed_date.month)[1]
                normalized_date = parsed_date.replace(day=last_day)
                return normalized_date.strftime("%Y-%m-%d")

        return None
    except Exception as e:
        return None

def get_filtered_meetings(start_date: Optional[str], end_date: Optional[str], selected_funds: Optional[List[str]]):
    """Get meeting notes filtered by date and funds."""
    try:
        client = MongoClient(MONGO_URI)
        try:
            db = client["Meetings"]
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
        finally:
            client.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting meetings: {e}")

def get_filtered_fund_comments(start_date: Optional[str], end_date: Optional[str], selected_funds: Optional[List[str]]):
    """Get factsheet count filtered by date and funds."""
    try:
        client = MongoClient(MONGO_URI)
        try:
            db = client["fund_reports"]
            collection = db["factsheets"]

            # Build query
            query = {}

            # Date filter - reportDate is stored as datetime
            if start_date and end_date:
                start_dt = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(end_date)
                query["reportDate"] = {
                    "$gte": start_dt,
                    "$lte": end_dt
                }

            # Fund filter - use fund_name field
            if selected_funds and len(selected_funds) > 0:
                query["fund_name"] = {"$in": selected_funds}

            # Count factsheets matching the query
            count = collection.count_documents(query)

            # Return a list with count elements to maintain compatibility with existing code
            return [{"count": i} for i in range(count)]
        finally:
            client.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting fund comments: {e}")

def get_filtered_transcripts(start_date: Optional[str], end_date: Optional[str], selected_funds: Optional[List[str]]):
    """Get transcripts filtered by date and funds."""
    try:
        client = MongoClient(MONGO_URI)
        try:
            db = client["Meetings"]
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

        finally:
            client.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting transcripts: {e}")

# ====================================================================================
# SMART RAG STRATEGY
# ====================================================================================

# Global cache for RAG results (in-memory, per-session)
rag_cache = {}

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

    # Check if filters changed
    if current_filters != previous_filters:
        print(f"[SCOPE] Filters changed: {previous_filters} -> {current_filters}")
        return True

    # Check if topic changed using LLM-based classification
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

def find_matching_fund_names(query: str, db) -> List[str]:
    """
    Find fund names that match the query (exact or partial).
    Returns list of matching fund_name values from MeetingChunks.
    """
    try:
        chunks_collection = db["MeetingChunks"]

        # Normalize query for matching - extract meaningful words
        query_lower = query.lower().strip()
        # Remove common question words to get the actual search terms
        stop_words = ['do', 'you', 'know', 'about', 'tell', 'me', 'what', 'is', 'are', 'the', 'a', 'an']
        query_words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]

        safe_query = query_lower.encode('ascii', 'ignore').decode('ascii')
        print(f"[FUND MATCH] Normalized query: {safe_query}".encode('ascii', 'ignore').decode('ascii'), flush=True)
        print(f"[FUND MATCH] Extracted search words: {query_words}".encode('ascii', 'ignore').decode('ascii'), flush=True)

        if not query_words:
            print(f"[FUND MATCH] No valid search words extracted".encode('ascii', 'ignore').decode('ascii'), flush=True)
            return []

        # Use aggregation to get unique fund names with text matching
        # Build regex pattern for any of the query words
        pattern = '|'.join(query_words)  # cooper|square

        matching_funds = []

        # Try aggregation approach for CosmosDB compatibility
        pipeline = [
            {"$match": {"fund_name": {"$regex": pattern, "$options": "i"}}},
            {"$group": {"_id": "$fund_name"}},
            {"$limit": 100}
        ]

        print(f"[FUND MATCH] Searching with pattern: {pattern}".encode('ascii', 'ignore').decode('ascii'), flush=True)

        results = list(chunks_collection.aggregate(pipeline))
        matching_funds = [r["_id"] for r in results if r["_id"] and isinstance(r["_id"], str)]

        print(f"[FUND MATCH] Found {len(matching_funds)} matching funds".encode('ascii', 'ignore').decode('ascii'), flush=True)

        for fund in matching_funds[:5]:  # Print first 5 matches
            safe_fund = fund.encode('ascii', 'ignore').decode('ascii')
            print(f"[FUND MATCH] Match: {safe_fund}".encode('ascii', 'ignore').decode('ascii'), flush=True)

        return matching_funds
    except Exception as e:
        error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
        print(f"[FUND MATCH] Error: {error_msg}".encode('ascii', 'ignore').decode('ascii'), flush=True)
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
        client = MongoClient(MONGO_URI)
        try:
            db = client["Meetings"]
            chunks_collection = db["MeetingChunks"]
            meetings_collection = db["MeetingNotes"]
            db_time = time.time() - db_start
            print(f"[TIME] DB Connection: {db_time:.3f}s")

            # STEP 0: Check if query matches any fund names (hybrid search boost)
            fund_match_start = time.time()
            matched_funds = find_matching_fund_names(question, db)
            if matched_funds and not selected_funds:
                # If we found matching fund names and no funds were explicitly selected,
                # boost results by adding fund filter
                print(f"[HYBRID SEARCH] Boosting search with matched funds: {matched_funds}")
                selected_funds = matched_funds
            fund_match_time = time.time() - fund_match_start
            print(f"[TIME] Fund Name Matching: {fund_match_time:.3f}s")

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

            # STEP 4: Execute vector search
            search_start = time.time()
            results = list(chunks_collection.aggregate(pipeline))
            search_time = time.time() - search_start
            print(f"[TIME] Vector Search Execution: {search_time:.3f}s")
            print(f"Found {len(results)} results")

            # Results are already limited by k in cosmosSearch
            print(f"Returning {len(results)} chunks")

            total_time = time.time() - total_start
            print(f"[TIME] TOTAL TIME: {total_time:.3f}s")
            print(f"[TIME] Breakdown - DB: {db_time:.3f}s | FundMatch: {fund_match_time:.3f}s | Embedding: {embedding_time:.3f}s | Filter: {filter_time:.3f}s | Pipeline: {pipeline_time:.3f}s | Search: {search_time:.3f}s")

            return results

        finally:
            client.close()

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
        client = MongoClient(MONGO_URI)
        try:
            db = client["fund_reports"]
            factsheets_collection = db["factsheets"]

            # Build filter query
            query = {}

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
            results = list(factsheets_collection.find(query).sort("reportDate", -1).limit(limit))

            elapsed = time.time() - start_time
            print(f"[FACTSHEET] Found {len(results)} factsheets in {elapsed:.3f}s")

            return results

        finally:
            client.close()

    except Exception as e:
        print(f"Error fetching factsheets: {str(e)}")
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
            else:
                factsheet_copy[key] = value

        # Convert to pretty-printed JSON
        context += f"Factsheet #{idx}:\n"
        context += json.dumps(factsheet_copy, indent=2, ensure_ascii=False)
        context += "\n\n" + "="*80 + "\n\n"

    return context

def ask_claude_with_rag(
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
                        top_k=100
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
                    top_k=100
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
        print(f"ERROR in ask_claude: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {e}")

def ask_claude_with_rag_streaming(
    question: str,
    data_sources: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    selected_funds: Optional[List[dict]] = None,
    conversation_history: Optional[List[dict]] = None
):
    """Send question with RAG-retrieved data to Gemini with streaming support."""
    try:
        print(f"[DEBUG] ask_claude_with_rag_streaming called with data_sources: {data_sources}")

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
                        top_k=100
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
                    top_k=100
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

        # Fetch factsheet data if selected
        if "Factsheet" in data_sources:
            print("[FACTSHEET] Fetching factsheet data...")
            factsheets = fetch_factsheets(
                start_date=start_date,
                end_date=end_date,
                selected_funds=selected_funds,
                limit=20
            )
            if factsheets:
                factsheet_context = format_factsheets_for_context(factsheets)
                context += factsheet_context
                print(f"[FACTSHEET] Added {len(factsheets)} factsheets to context")
            else:
                print("[FACTSHEET] No factsheets found")

        # Web Search will be handled by Google Search Grounding tool (configured below)
        use_google_search = "Web Search" in data_sources
        if use_google_search:
            print("[WEB SEARCH] Google Search Grounding enabled")

        # Debug: Print context length being sent to Claude
        print(f"DEBUG - Total context length: {len(context)} characters")
        # Skip context preview to avoid encoding issues with special characters
        # print(f"DEBUG - Context preview (first 500 chars):\n{context[:500]}")

        # Build prompt with conversation history
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
   - **CITATION REQUIREMENTS**:
     * Cite EVERY point from internal data with specific sources (e.g., "Meeting with John Doe, Sept 15, 2025", "Q3 2025 Factsheet")
     * Use inline citations throughout your analysis, not just at the end
     * Make citations specific and detailed so users know exactly where information came from
   - **STRUCTURE YOUR RESPONSE IN TWO CLEAR SECTIONS**:
     * **Section 1 - Internal Data Analysis**: Present all information from meeting notes and factsheets with detailed inline citations
     * **Section 2 - Web Search Results**: Present additional current information found via web search with titles and URLs
     * Include both analysis AND sources in the web search section - not just links
   - Web searches MUST be relevant to entities found in internal data
"""
        else:
            web_search_instructions = """
3. Only use the internal data provided below. Web search is not enabled.
"""

        full_prompt += f"""Question: {question}

Instructions:
1. Read and analyze the internal data sources provided below (meeting notes and factsheets)
2. Present relevant internal information with comprehensive details
{web_search_instructions}
4. Synthesize all information (internal + web if enabled) into one comprehensive answer
5. Provide thorough explanations with context, background, and analysis
6. Cite all sources with specifics (meeting dates, URLs, fund names, manager names)
7. Use clear section headers to distinguish internal data from web search results
8. Give detailed, comprehensive responses - do not be brief

Available data sources: {', '.join(data_sources)}

Internal Data:
{context}

Now provide a comprehensive, detailed answer to the question above."""

        # Debug: Print the final prompt being sent to Gemini
        print(f"DEBUG - Final prompt length: {len(full_prompt)} characters")
        print(f"DEBUG - Final prompt preview (first 1000 chars):\n{full_prompt[:1000]}")

        # Configure Google Search Grounding if web search is enabled
        config = None
        from google.genai import types

        if use_google_search:
            # Use Google Search Grounding - Gemini automatically searches, reads, and synthesizes web content
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(
                tools=[google_search_tool],
                temperature=1  # Maximum creativity and varied responses
            )
            print("[WEB SEARCH] Configured Google Search Grounding (automatic web content analysis)")
        else:
            # Configure for comprehensive responses with no length restrictions
            config = types.GenerateContentConfig(
                temperature=1
            )

        # Send to Gemini with streaming
        if config:
            response = gemini_client.models.generate_content_stream(
                model="gemini-3-pro-preview",
                contents=full_prompt,
                config=config
            )
        else:
            response = gemini_client.models.generate_content_stream(
                model="gemini-3-pro-preview",
                contents=full_prompt
            )

        for chunk in response:
            try:
                if hasattr(chunk, 'text') and chunk.text:
                    print(f"[STREAMING] Yielding chunk: {len(chunk.text)} chars")
                    yield chunk.text
                else:
                    print(f"[STREAMING] Chunk has no text attribute or empty text")
            except Exception as chunk_error:
                print(f"[STREAMING ERROR] Error processing chunk: {chunk_error}")
                # Try to extract text from candidates if direct access fails
                try:
                    if hasattr(chunk, 'candidates'):
                        for candidate in chunk.candidates:
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        print(f"[STREAMING] Extracted text from parts: {len(part.text)} chars")
                                        yield part.text
                except Exception as extract_error:
                    print(f"[STREAMING ERROR] Failed to extract text: {extract_error}")

    except Exception as e:
        print(f"ERROR in ask_claude_streaming: {str(e)}")
        import traceback
        traceback.print_exc()
        yield f"Error: {str(e)}"

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
        client = MongoClient(MONGO_URI)
        try:
            all_portfolios = set()

            # Get LODH portfolios
            lodh_db = client["LODH"]
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
            pictet_db = client["Pictet"]
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

        finally:
            client.close()

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
            client = MongoClient(MONGO_URI)
            try:
                # Get LODH funds
                lodh_db = client["LODH"]
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
                pictet_db = client["Pictet"]
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

            finally:
                client.close()

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
        client = MongoClient(MONGO_URI)
        try:
            # Get funds from meeting notes
            meetings_db = client["Meetings"]
            meetings_collection = meetings_db["Meeting Notes"]
            meeting_funds = meetings_collection.distinct("UniqueName")

            # Get funds from fund reports
            reports_db = client["fund_reports"]
            reports_collection = reports_db["factsheets"]
            report_funds = reports_collection.distinct("UniqueName")

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
        finally:
            client.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting funds: {e}")

@app.post("/api/chat/stats", response_model=FilterStats)
async def get_filter_stats(request: ChatRequest):
    """Get counts based on current filters."""
    try:
        meetings_count = 0
        comments_count = 0
        transcripts_count = 0

        if "Meeting Notes" in request.data_sources:
            meetings = get_filtered_meetings(request.start_date, request.end_date, request.selected_funds)
            meetings_count = len(meetings)

        if "Factsheet" in request.data_sources:
            comments = get_filtered_fund_comments(request.start_date, request.end_date, request.selected_funds)
            comments_count = len(comments)

        if "Transcripts" in request.data_sources:
            transcripts = get_filtered_transcripts(request.start_date, request.end_date, request.selected_funds)
            transcripts_count = len(transcripts)

        return FilterStats(
            meeting_notes_count=meetings_count,
            factsheet_comments_count=comments_count,
            transcripts_count=transcripts_count,
            total_count=meetings_count + comments_count + transcripts_count
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

        answer = ask_claude_with_rag(
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

        def generate():
            try:
                print("[DEBUG] Starting generate() function")
                for item in ask_claude_with_rag_streaming(
                    question=request.question,
                    data_sources=request.data_sources,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    selected_funds=request.selected_funds,
                    conversation_history=request.conversation_history
                ):
                    print(f"[DEBUG] Received item from streaming: {type(item)}")
                    # Check if it's a status tuple or text content
                    if isinstance(item, tuple) and item[0] == 'STATUS':
                        # Send search status event
                        yield f"data: {json.dumps({'searching': item[1]})}\n\n"
                    else:
                        # Send each text chunk as Server-Sent Events format
                        yield f"data: {json.dumps({'content': item})}\n\n"

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
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
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

        client = MongoClient(MONGO_URI)
        try:
            db = client["Chatbot"]
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
        finally:
            client.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {e}")

@app.get("/api/chat/conversations")
async def list_conversations(request: Request, session_id: Optional[str] = None):
    """List all conversations for the current user."""
    try:
        user_id = get_user_identifier(request, session_id)

        client = MongoClient(MONGO_URI)
        try:
            db = client["Chatbot"]
            collection = db["Conversations"]

            # Find all conversations for this user, sorted by updated_at
            conversations = list(collection.find(
                {"user_id": user_id},
                {"_id": 0, "conversation_id": 1, "title": 1, "created_at": 1, "updated_at": 1, "messages": 1}
            ).sort("updated_at", -1))

            # Convert datetime objects to strings and add message count
            for conv in conversations:
                conv["created_at"] = conv["created_at"].isoformat() if "created_at" in conv else None
                conv["updated_at"] = conv["updated_at"].isoformat() if "updated_at" in conv else None
                conv["message_count"] = len(conv.get("messages", []))
                # Remove messages from list view for performance
                conv.pop("messages", None)

            return {
                "success": True,
                "conversations": conversations
            }
        finally:
            client.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {e}")

@app.get("/api/chat/conversations/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str, session_id: Optional[str] = None):
    """Get a specific conversation."""
    try:
        user_id = get_user_identifier(request, session_id)

        client = MongoClient(MONGO_URI)
        try:
            db = client["Chatbot"]
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
        finally:
            client.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {e}")

@app.post("/api/chat/conversations/{conversation_id}/messages")
async def save_message(request: Request, conversation_id: str, chat_request: SaveChatRequest):
    """Save a message to a conversation."""
    try:
        user_id = get_user_identifier(request, chat_request.session_id)

        client = MongoClient(MONGO_URI)
        try:
            db = client["Chatbot"]
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
        finally:
            client.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving message: {e}")

@app.delete("/api/chat/conversations/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str, session_id: Optional[str] = None):
    """Delete a conversation."""
    try:
        user_id = get_user_identifier(request, session_id)

        client = MongoClient(MONGO_URI)
        try:
            db = client["Chatbot"]
            collection = db["Conversations"]

            result = collection.delete_one({"conversation_id": conversation_id, "user_id": user_id})

            if result.deleted_count == 0:
                raise HTTPException(status_code=404, detail="Conversation not found")

            return {
                "success": True,
                "message": "Conversation deleted successfully"
            }
        finally:
            client.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {e}")

@app.get("/api/debug/test-cooper")
async def test_cooper_search():
    """Debug endpoint to test Cooper Square search"""
    try:
        client = MongoClient(MONGO_URI)
        db = client["Sterwen"]
        chunks = db["MeetingChunks"]

        # Test 1: Exact match
        exact_count = chunks.count_documents({"fund_name": "Cooper Square"})

        # Test 2: Case-insensitive regex
        regex_results = list(chunks.find({"fund_name": {"$regex": "cooper", "$options": "i"}}).limit(5))
        regex_funds = [doc.get("fund_name") for doc in regex_results]

        # Test 3: Aggregation
        pipeline = [
            {"$match": {"fund_name": {"$regex": "cooper", "$options": "i"}}},
            {"$group": {"_id": "$fund_name"}},
            {"$limit": 10}
        ]
        agg_results = list(chunks.aggregate(pipeline))
        agg_funds = [r["_id"] for r in agg_results]

        client.close()

        return {
            "exact_match_count": exact_count,
            "regex_find_results": regex_funds,
            "aggregation_results": agg_funds,
            "test_completed": True
        }
    except Exception as e:
        return {"error": str(e), "test_completed": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

