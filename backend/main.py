"""
Meeting Notes & Fund Comments Chatbot API
==========================================
FastAPI backend for intelligent Q&A system with meeting notes, factsheet comments, and transcripts
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date
from pymongo import MongoClient
import calendar
import os
import json
import anthropic
from dotenv import load_dotenv
from embedding_service import EmbeddingService

# Load environment variables
load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://mongodbsterwen:Genevaboy$1204@sterwendb.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Debug: Check if API key is loaded
print(f"CLAUDE_API_KEY loaded: {bool(CLAUDE_API_KEY)} (length: {len(CLAUDE_API_KEY) if CLAUDE_API_KEY else 0})")
print("=" * 80)
print("🚀 CONTAINER VERSION: 2024-11-10-NGINX-FIXED 🚀")
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

# Helper Functions
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
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        try:
            db = client["Meetings"]
            chunks_collection = db["MeetingChunks"]
            meetings_collection = db["MeetingNotes"]

            # STEP 1: Build MongoDB filter to narrow search space BEFORE vector search
            match_filter = {}

            # Add date range filter
            if start_date and end_date:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                match_filter["meeting_date"] = {
                    "$gte": start_date_obj,
                    "$lte": end_date_obj
                }
                print(f"Pre-filtering by date: {start_date} to {end_date}")

            # Add fund filter
            if selected_funds and len(selected_funds) > 0:
                match_filter["fund_name"] = {"$in": selected_funds}
                print(f"Pre-filtering by funds: {selected_funds}")

            # STEP 2: Generate embedding
            embedding_service = EmbeddingService(GEMINI_API_KEY)
            print(f"Generating embedding for query: {question[:100]}...")
            query_embedding = embedding_service.generate_embedding(
                question,
                task_type="RETRIEVAL_QUERY"
            )

            # STEP 3: Build pipeline with $match BEFORE $search to filter first
            pipeline = []

            # Add $match stage first to filter documents before vector search
            if match_filter:
                pipeline.append({"$match": match_filter})
                print(f"Added pre-filter stage: {match_filter}")

            # Add vector search stage (searches only filtered documents)
            pipeline.extend([
                {
                    "$search": {
                        "cosmosSearch": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "k": 300
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
            ])

            # Execute pipeline
            print(f"Executing filtered vector search pipeline...")
            results = list(chunks_collection.aggregate(pipeline))
            print(f"Vector search on filtered data returned {len(results)} chunks")

            # STEP 4: Limit to top_k
            results = results[:top_k]
            print(f"Returning top {len(results)} chunks")

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
    """Send question with RAG-retrieved data to Claude."""
    try:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

        # Prepare context
        context = "Available Data Sources:\n\n"

        # Use RAG for meeting notes
        if "Meeting Notes" in data_sources:
            print("Using RAG vector search for meeting notes...")
            rag_chunks = perform_vector_search(
                question=question,
                data_sources=data_sources,
                start_date=start_date,
                end_date=end_date,
                selected_funds=selected_funds,
                top_k=50
            )

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

        # Build messages with conversation history
        messages = []

        if conversation_history:
            for item in conversation_history[-3:]:  # Last 3 exchanges
                messages.append({
                    "role": "user",
                    "content": item.get('question', '')
                })
                messages.append({
                    "role": "assistant",
                    "content": item.get('answer', '')
                })

        # Add current question with context
        current_prompt = f"""Based on the following data sources, please answer this question: {question}

Important instructions:
- Use information from the provided data sources below AND from our conversation history
- If the question refers to something mentioned earlier in our conversation, you can use that information
- If you cannot find relevant information in either the data or conversation history, say so clearly
- Cite specific sources when possible (meeting dates, report dates, fund names)
- Be concise and factual
- Distinguish between meeting notes, monthly factsheet comments, and meeting transcripts in your response

Available data sources: {', '.join(data_sources)}

{context}"""

        messages.append({
            "role": "user",
            "content": current_prompt
        })

        # Send to Claude (non-streaming version for backward compatibility)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=messages
        )

        return response.content[0].text.strip()

    except Exception as e:
        print(f"ERROR in ask_claude: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error calling Claude API: {e}")

def ask_claude_with_rag_streaming(
    question: str,
    data_sources: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    selected_funds: Optional[List[str]] = None,
    conversation_history: Optional[List[dict]] = None
):
    """Send question with RAG-retrieved data to Claude with streaming support."""
    try:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

        # Prepare context (same as non-streaming version)
        context = "Available Data Sources:\n\n"

        # Use RAG for meeting notes
        if "Meeting Notes" in data_sources:
            print("Using RAG vector search for meeting notes...")
            rag_chunks = perform_vector_search(
                question=question,
                data_sources=data_sources,
                start_date=start_date,
                end_date=end_date,
                selected_funds=selected_funds,
                top_k=50
            )

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

        # Build messages with conversation history
        messages = []

        if conversation_history:
            for item in conversation_history[-3:]:  # Last 3 exchanges
                messages.append({
                    "role": "user",
                    "content": item.get('question', '')
                })
                messages.append({
                    "role": "assistant",
                    "content": item.get('answer', '')
                })

        # Add current question with context
        current_prompt = f"""Based on the following data sources, please answer this question: {question}

Important instructions:
- Use information from the provided data sources below AND from our conversation history
- If the question refers to something mentioned earlier in our conversation, you can use that information
- If you cannot find relevant information in either the data or conversation history, say so clearly
- Cite specific sources when possible (meeting dates, report dates, fund names)
- Be concise and factual
- Distinguish between meeting notes, monthly factsheet comments, and meeting transcripts in your response

Available data sources: {', '.join(data_sources)}

{context}"""

        messages.append({
            "role": "user",
            "content": current_prompt
        })

        # Send to Claude with streaming
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text

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
                for text_chunk in ask_claude_with_rag_streaming(
                    question=request.question,
                    data_sources=request.data_sources,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    selected_funds=request.selected_funds,
                    conversation_history=request.conversation_history
                ):
                    # Send each chunk as Server-Sent Events format
                    yield f"data: {json.dumps({'content': text_chunk})}\n\n"

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
