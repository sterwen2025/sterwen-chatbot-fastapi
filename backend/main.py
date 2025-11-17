"""
Meeting Notes & Fund Comments Chatbot API
==========================================
FastAPI backend for intelligent Q&A system with meeting notes, factsheet comments, and transcripts
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
from dotenv import load_dotenv
from embedding_service import EmbeddingService
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://mongodbsterwen:Genevaboy$1204@sterwendb.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

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

def perform_web_search(query: str, max_results: int = 5):
    """
    Perform web search using DuckDuckGo.
    Returns list of search results with title, snippet, and URL.
    """
    try:
        print(f"[WEB SEARCH] Searching for: {query}")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": result.get("href", "")
                })
            print(f"[WEB SEARCH] Found {len(formatted_results)} results")
            return formatted_results
    except Exception as e:
        print(f"[WEB SEARCH] Error: {e}")
        return []

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
    """Get fund comments filtered by date and funds."""
    try:
        client = MongoClient(MONGO_URI)
        try:
            db = client["fund_reports"]
            collection = db["factsheets"]

            # Build query
            query_conditions = []

            # Fund filter
            if selected_funds:
                fund_condition = {"UniqueName": {"$in": selected_funds}}
                query_conditions.append(fund_condition)

            # Only get documents with non-empty comments
            comment_condition = {
                "$and": [
                    {"comment": {"$exists": True}},
                    {"comment": {"$ne": ""}},
                    {"comment": {"$ne": None}},
                    {"comment": {"$ne": "None"}}
                ]
            }
            query_conditions.append(comment_condition)

            # Combine conditions
            if query_conditions:
                query = {"$and": query_conditions}
            else:
                query = {"comment": {"$exists": True, "$ne": "", "$ne": None, "$ne": "None"}}

            # Get all comments
            comments = list(collection.find(query, {
                "fund_name": 1,
                "UniqueName": 1,
                "reportDate": 1,
                "comment": 1,
                "_id": 0
            }))

            # Normalize dates and filter
            filtered_comments = []
            for comment in comments:
                original_date = comment.get('reportDate')
                normalized_date = normalize_date_to_month_end(original_date)

                if normalized_date:
                    comment['reportDate'] = normalized_date

                    # Apply date filter
                    if start_date and end_date:
                        comment_date_obj = datetime.strptime(normalized_date, "%Y-%m-%d").date()
                        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
                        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
                        if start_date_obj <= comment_date_obj <= end_date_obj:
                            filtered_comments.append(comment)
                    else:
                        filtered_comments.append(comment)
                elif not start_date and not end_date:
                    filtered_comments.append(comment)

            return filtered_comments
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
- SAME topic: Follow-up questions, requests for more details, clarifications, or elaborations about the same subject/entity
  Examples: "Tell me about AQR" → "More details" (SAME)
               "What is their strategy?" → "Can you elaborate?" (SAME)
- DIFFERENT topic: Questions about different entities, companies, funds, or completely different subjects
  Examples: "Tell me about AQR" → "Tell me about FengHe" (DIFFERENT)
               "What is AQR's strategy?" → "Tell me about Citadel" (DIFFERENT)

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
    previous_scope = rag_cache.get(cache_key, {}).get('scope')

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
            print(f"[TIME] Breakdown - DB: {db_time:.3f}s | Embedding: {embedding_time:.3f}s | Filter: {filter_time:.3f}s | Pipeline: {pipeline_time:.3f}s | Search: {search_time:.3f}s")

            return results

        finally:
            client.close()

    except Exception as e:
        print(f"Error in vector search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing vector search: {e}")

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

        # TODO: Implement RAG for factsheet comments and transcripts later
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
- Use information from the provided data sources below AND from our conversation history
- If the question refers to something mentioned earlier in our conversation, you can use that information
- **ALWAYS present all Web Search Results when they are provided in the data sources section**
- When web search results are provided, present them as your answer - do not say there is no information
- List web search results with their titles and clickable URLs
- Do NOT add disclaimers about missing information if web search results are available
- Cite specific sources when possible (meeting dates, report dates, fund names, URLs for web search results)
- Be concise and factual
- Distinguish between meeting notes, monthly factsheet comments, meeting transcripts, and web search results in your response

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
    selected_funds: Optional[List[str]] = None,
    conversation_history: Optional[List[dict]] = None
):
    """Send question with RAG-retrieved data to Gemini with streaming support."""
    try:

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

        # Add Web Search results if selected
        if "Web Search" in data_sources:
            print("[WEB SEARCH] Performing web search...")
            yield ('STATUS', True)
            web_results = perform_web_search(question, max_results=5)
            yield ('STATUS', False)

            if web_results:
                context += "\n=== WEB SEARCH RESULTS ===\n\n"
                for idx, result in enumerate(web_results, 1):
                    context += f"{idx}. {result['title']}\n"
                    context += f"   URL: {result['url']}\n"
                    context += f"   {result['snippet']}\n\n"
                    print(f"[WEB SEARCH] Result {idx}: {result['title'][:100]}")
                print(f"Added {len(web_results)} web search results to context")
                print(f"[WEB SEARCH] Context preview:\n{context[context.find('=== WEB SEARCH RESULTS ==='):context.find('=== WEB SEARCH RESULTS ===')+500]}")
            else:
                print("No web search results found")

        # Debug: Print context length being sent to Claude
        print(f"DEBUG - Total context length: {len(context)} characters")
        print(f"DEBUG - Context preview (first 500 chars):\n{context[:500]}")

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
- Use information from the provided data sources below AND from our conversation history
- If the question refers to something mentioned earlier in our conversation, you can use that information
- **ALWAYS present all Web Search Results when they are provided in the data sources section**
- When web search results are provided, present them as your answer - do not say there is no information
- List web search results with their titles and clickable URLs
- Do NOT add disclaimers about missing information if web search results are available
- Cite specific sources when possible (meeting dates, report dates, fund names, URLs for web search results)
- Be concise and factual
- Distinguish between meeting notes, monthly factsheet comments, meeting transcripts, and web search results in your response

Available data sources: {', '.join(data_sources)}

{context}"""

        # Debug: Print the final prompt being sent to Gemini
        print(f"DEBUG - Final prompt length: {len(full_prompt)} characters")
        print(f"DEBUG - Final prompt preview (first 1000 chars):\n{full_prompt[:1000]}")

        # Send to Gemini with streaming
        response = gemini_client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=full_prompt
        )

        for chunk in response:
            yield chunk.text

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

        if "Factsheet Comments" in request.data_sources:
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
                "factsheet_comments": "Filtered" if "Factsheet Comments" in request.data_sources else 0,
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
                for item in ask_claude_with_rag_streaming(
                    question=request.question,
                    data_sources=request.data_sources,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    selected_funds=request.selected_funds,
                    conversation_history=request.conversation_history
                ):
                    # Check if it's a status tuple or text content
                    if isinstance(item, tuple) and item[0] == 'STATUS':
                        # Send search status event
                        yield f"data: {json.dumps({'searching': item[1]})}\n\n"
                    else:
                        # Send each text chunk as Server-Sent Events format
                        yield f"data: {json.dumps({'content': item})}\n\n"

                # Send done signal
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                print(f"Error in generate: {str(e)}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
