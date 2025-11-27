# Chatbot RAG Implementation Documentation

## Overview
This document details the RAG (Retrieval Augmented Generation) implementation for the meeting notes chatbot, which provides intelligent Q&A over meeting notes, factsheets, and web search results.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE (React)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Data Source  │  │ Date Filter  │  │ Fund Filter  │  │ Model Selection      │ │
│  │ Selection    │  │ (Optional)   │  │ (Optional)   │  │ (Flash/Pro/Thinking) │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│                                    │                                             │
│                           ┌────────▼────────┐                                    │
│                           │   Question Box   │                                   │
│                           │  + Stop Button   │                                   │
│                           └────────┬─────────┘                                   │
└────────────────────────────────────┼─────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND (FastAPI + Python)                             │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 1: SMART RAG STRATEGY                            │    │
│  │  ┌──────────────┐    ┌─────────────────┐    ┌────────────────────────┐  │    │
│  │  │ Intent       │───▶│ Scope Change    │───▶│ Strategy Decision      │  │    │
│  │  │ Classification│    │ Detection (LLM) │    │ NO_RAG/REUSE/NEW_QUERY│  │    │
│  │  └──────────────┘    └─────────────────┘    └────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 2: HYBRID SEARCH (if needed)                     │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │                    MEETING NOTES SEARCH                          │    │    │
│  │  │  ┌────────────────┐          ┌────────────────┐                 │    │    │
│  │  │  │ Vector Search  │          │ BM25 Keyword   │                 │    │    │
│  │  │  │ (Semantic)     │    ║     │ Search         │                 │    │    │
│  │  │  │ Gemini Embed   │    ║     │ (Exact Match)  │                 │    │    │
│  │  │  └───────┬────────┘    ║     └───────┬────────┘                 │    │    │
│  │  │          └─────────────╬─────────────┘                          │    │    │
│  │  │                        ║                                         │    │    │
│  │  │              ┌─────────▼─────────┐                              │    │    │
│  │  │              │ RRF Fusion        │                              │    │    │
│  │  │              │ (50/50 weighting) │                              │    │    │
│  │  │              └─────────┬─────────┘                              │    │    │
│  │  └────────────────────────┼────────────────────────────────────────┘    │    │
│  │                           │                                              │    │
│  │  ┌────────────────────────▼────────────────────────────────────────┐    │    │
│  │  │                    FACTSHEET SEARCH                              │    │    │
│  │  │  ┌────────────────┐          ┌────────────────┐                 │    │    │
│  │  │  │ Vector Search  │          │ BM25 Keyword   │                 │    │    │
│  │  │  │ (Semantic)     │    ║     │ Search         │                 │    │    │
│  │  │  └───────┬────────┘    ║     └───────┬────────┘                 │    │    │
│  │  │          └─────────────╬─────────────┘                          │    │    │
│  │  │              ┌─────────▼─────────┐                              │    │    │
│  │  │              │ RRF Fusion        │                              │    │    │
│  │  │              └───────────────────┘                              │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 3: CONTEXT BUILDING                              │    │
│  │  • Format meeting note chunks with metadata                              │    │
│  │  • Format factsheet chunks with deduplication                           │    │
│  │  • Include conversation history (last 3 exchanges)                       │    │
│  │  • Build comprehensive prompt with instructions                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 4: GEMINI API CALL                               │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │    │
│  │  │ gemini-2.5-  │    │ gemini-2.5-  │    │ gemini-3-pro-preview     │  │    │
│  │  │ flash        │    │ pro          │    │ (with thinking config)   │  │    │
│  │  └──────────────┘    └──────────────┘    └──────────────────────────┘  │    │
│  │                                                                          │    │
│  │  Optional: Google Search Grounding (if Web Search enabled)              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    STEP 5: STREAMING RESPONSE                            │    │
│  │  • Server-Sent Events (SSE) to frontend                                  │    │
│  │  • Status updates: "Finding relevant information..."                     │    │
│  │  • Status updates: "Generating response..."                              │    │
│  │  • Thinking summaries (for Gemini 3 thinking models)                    │    │
│  │  • Content chunks streamed in real-time                                  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA STORES                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────────────┐ │
│  │ Azure Cosmos DB      │  │ Azure Cosmos DB      │  │ Google Search         │ │
│  │ MeetingChunks        │  │ FactsheetChunks      │  │ Grounding API         │ │
│  │ (768-dim vectors)    │  │ (768-dim vectors)    │  │ (Real-time web data)  │ │
│  └──────────────────────┘  └──────────────────────┘  └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Logic Flow

### 1. User Input Processing

When a user sends a question, the frontend collects:
- **Question text**: The user's natural language query
- **Data sources**: Which databases to search (Meeting Notes, Factsheet, Web Search)
- **Date filter** (optional): Start and end dates to narrow results
- **Fund filter** (optional): Specific fund names to search
- **Model selection**: Which Gemini model to use

The frontend also provides:
- **Stop button**: Allows user to abort request mid-stream using AbortController
- **Status indicators**: Shows "Finding relevant information..." and "Generating response..."

### 2. Smart RAG Strategy (Optimization Layer)

Before performing expensive vector searches, the system determines the optimal strategy:

#### 2.1 Intent Classification
```python
def classify_intent(question, conversation_history):
    # Transform patterns - don't need new search
    transform_patterns = ['summarize', 'explain', 'elaborate', 'tell me more', ...]

    if any(pattern in question.lower() for pattern in transform_patterns):
        if conversation_history and len(conversation_history) >= 2:
            return 'NO_RAG'  # Skip search, use existing context

    return 'NEW_QUERY'  # Need new search
```

#### 2.2 Scope Change Detection (LLM-based)
```python
def detect_scope_change(question, filters, previous_scope):
    # Quick check: if filters changed, skip LLM call
    if filters_changed:
        return True

    # Use Gemini to classify topic change
    prompt = f"""
    Previous question: "{previous_question}"
    Current question: "{question}"

    Are these about SAME or DIFFERENT topics?
    """

    response = gemini_client.generate_content(prompt)
    return response == "DIFFERENT"
```

#### 2.3 Strategy Decision
| Scenario | Strategy | Action |
|----------|----------|--------|
| Follow-up question (summarize, explain) | NO_RAG | Skip search, use conversation context |
| Same topic, same filters | RAG_REUSE | Use cached search results |
| New topic or changed filters | RAG_NEW_QUERY | Perform new vector search |

### 3. Hybrid Search System

The system uses a **hybrid search** approach combining:

#### 3.1 Vector Search (Semantic Understanding)
```python
def perform_vector_search(question, filters, top_k=50):
    # Step 1: Generate query embedding
    embedding_service = EmbeddingService(GEMINI_API_KEY)
    query_embedding = embedding_service.generate_embedding(
        question,
        task_type="RETRIEVAL_QUERY"
    )

    # Step 2: Build filter for Cosmos DB
    cosmos_filter = {}
    if date_filter:
        cosmos_filter["meeting_date"] = {"$gte": start, "$lte": end}
    if fund_filter:
        cosmos_filter["fund_name"] = {"$in": selected_funds}

    # Step 3: Execute vector search with pre-filtering
    pipeline = [
        {
            "$search": {
                "cosmosSearch": {
                    "vector": query_embedding,
                    "path": "embedding",
                    "k": top_k,
                    "filter": cosmos_filter,
                    "exact": True  # ENN for filtered searches
                }
            }
        }
    ]

    return chunks_collection.aggregate(pipeline)
```

#### 3.2 BM25 Search (Keyword Matching)
```python
class BM25:
    """
    Best Matching 25 algorithm for exact keyword matching.
    Complements vector search by finding exact entity names.
    """
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Length normalization

    def search(self, query, top_k=50):
        # Calculate BM25 score for each document
        # IDF * TF / (TF + k1 * (1 - b + b * doc_len/avg_len))
        return sorted_results[:top_k]
```

#### 3.3 Reciprocal Rank Fusion (RRF)
```python
def merge_vector_and_bm25_results(vector_results, bm25_results, k=60):
    """
    Merge two ranked lists using RRF.
    RRF_score = sum(1 / (k + rank)) for each list
    """
    rrf_scores = {}

    # Add vector scores (50% weight)
    for rank, result in enumerate(vector_results):
        rrf_scores[result.id] += 0.5 * (1 / (k + rank + 1))

    # Add BM25 scores (50% weight)
    for rank, result in enumerate(bm25_results):
        rrf_scores[result.id] += 0.5 * (1 / (k + rank + 1))

    return sorted(results, key=lambda x: rrf_scores[x.id], reverse=True)
```

### 4. Data Source Processing

#### 4.1 Meeting Notes
- **Source**: MeetingChunks collection (Azure Cosmos DB)
- **Fields**: meeting_date, fund_name, manager, chunk_type, text, embedding
- **Search**: Hybrid (Vector + BM25) executed in parallel using ThreadPoolExecutor

#### 4.2 Factsheets
- **Source**: FactsheetChunks collection (Azure Cosmos DB)
- **Fields**: report_date, UniqueName, fund_name, section_type, text, embedding, full_document
- **Search**: Hybrid (Vector + BM25)
- **Deduplication**: Groups chunks by factsheet_id to avoid showing same factsheet multiple times

#### 4.3 Web Search
- **Source**: Google Search Grounding API (built into Gemini)
- **Activation**: When "Web Search" is selected as data source
- **Behavior**: Gemini automatically searches, reads, and synthesizes web content

### 5. Context Building

```python
def build_context(rag_chunks, factsheet_chunks, conversation_history):
    context = "Available Data Sources:\n\n"

    # Add meeting notes
    if rag_chunks:
        context += "=== MEETING NOTES (RAG Retrieved) ===\n\n"
        for chunk in rag_chunks:
            context += f"Date: {chunk['meeting_date']}\n"
            context += f"Fund: {chunk['fund_name']}\n"
            context += f"Manager: {chunk['manager']}\n"
            context += f"Content: {chunk['text']}\n"
            context += f"Relevance Score: {chunk['score']:.4f}\n"
            context += "\n---\n\n"

    # Add factsheets (deduplicated)
    if factsheet_chunks:
        context += format_factsheet_chunks_for_context(factsheet_chunks)

    return context
```

### 6. Prompt Engineering

The system builds a comprehensive prompt with specific instructions:

```python
full_prompt = f"""Question: {question}

Instructions:
1. **FORMATTING**: Use ONLY standard markdown - NO HTML tags
2. **DATE AWARENESS**: Always state the date of data being referenced
3. **INTERNAL DATA PRIORITY**: Read and analyze meeting notes and factsheets first
4. **WEB SEARCH** (if enabled):
   - Use context from internal data to make searches relevant
   - Cite all web sources with clickable markdown links
   - Structure response with Section 1 (Internal) and Section 2 (Web)
5. **CITATION STYLE**: Use fund name and date, NEVER generic "Factsheet 1"

Available data sources: {data_sources}

Internal Data:
{context}

Now provide a comprehensive, detailed answer."""
```

### 7. Gemini Model Selection

| Model | Use Case | Configuration |
|-------|----------|---------------|
| gemini-2.5-flash | Fast responses | Default |
| gemini-2.5-pro | Complex analysis | Default |
| gemini-3-pro-preview (Low) | Thoughtful answers | thinking_level="low" |
| gemini-3-pro-preview (High) | Deep reasoning | Default thinking |

### 8. Streaming Response

```python
def ask_gemini_with_rag_streaming(...):
    # Send status: searching
    yield ('STATUS', True)

    # Perform searches...

    # Send status: generating
    yield ('GENERATING', True)

    # Stream response
    response = gemini_client.generate_content_stream(...)
    for chunk in response:
        # Handle thinking summaries (Gemini 3)
        if chunk.thought:
            yield ('THINKING', chunk.text)
        # Handle content
        elif chunk.text:
            yield chunk.text
```

### 9. Frontend Display

The frontend handles streaming events:

```typescript
for (const line of lines) {
    const jsonData = JSON.parse(line.slice(6));

    if (jsonData.searching) {
        setSearchStatus('Finding relevant information...');
    }
    if (jsonData.generating) {
        setSearchStatus('Generating response...');
    }
    if (jsonData.thinking) {
        setThinkingSummary(jsonData.thinking);
    }
    if (jsonData.content) {
        accumulatedAnswer += jsonData.content;
        // Update UI with streamed content
    }
}
```

---

## Database Schema

### MeetingChunks Collection
```json
{
  "chunk_id": "meeting-123-chunk-0",
  "meeting_id": "meeting-123",
  "chunk_index": 0,
  "chunk_type": "notes_part" | "conclusion" | "full_notes",
  "text": "The meeting notes content...",
  "embedding": [0.123, 0.456, ...],  // 768-dimensional vector
  "fund_name": "Example Fund",
  "manager": "John Doe",
  "meeting_date": "2024-01-15",
  "embedding_model": "gemini-embedding-001"
}
```

### FactsheetChunks Collection
```json
{
  "chunk_id": "factsheet-abc-chunk-0",
  "factsheet_id": "factsheet-abc",
  "section_type": "performance" | "holdings" | "commentary",
  "text": "The factsheet section content...",
  "embedding": [0.123, 0.456, ...],
  "UniqueName": "Example Fund",
  "fund_name": "Example Fund",
  "report_date": "2024-01-31T00:00:00Z",
  "full_document": { /* complete factsheet JSON */ }
}
```

---

## Performance Characteristics

### Typical Response Times
| Phase | Time |
|-------|------|
| Embedding Generation | 0.3-0.5s |
| Vector Search | 1-2s |
| BM25 Search | 0.5-1s |
| Context Building | 0.1s |
| Gemini API (first token) | 1-3s |
| Total (with search) | 3-7s |
| Total (NO_RAG) | 1-3s |

### Optimization Strategies
1. **Parallel Execution**: Vector and BM25 searches run simultaneously
2. **RAG Caching**: Reuse search results for follow-up questions
3. **Intent Classification**: Skip unnecessary searches
4. **Connection Pooling**: Reuse MongoDB connections (maxPoolSize=50)
5. **ENN Search**: Use exact nearest neighbor for filtered searches (50% faster)

---

## Frontend Features

### Stop Button Implementation
```typescript
const abortControllerRef = useRef<AbortController | null>(null);

const handleStop = () => {
    if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        setLoading(false);
        message.info('Request stopped');
    }
};

// In fetch call
const abortController = new AbortController();
abortControllerRef.current = abortController;
fetch(url, { signal: abortController.signal });
```

### Status Display
- **Finding relevant information...**: Shown during vector/BM25 search
- **Generating response...**: Shown while waiting for Gemini
- **Thinking...** (with summary): Shown for Gemini 3 thinking models

---

## API Endpoints

### POST /api/chat/ask/stream
Main streaming endpoint for chat.

**Request:**
```json
{
  "question": "What did AQR discuss in their last meeting?",
  "data_sources": ["Meeting Notes", "Factsheet", "Web Search"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "selected_funds": ["AQR Momentum"],
  "conversation_history": [...],
  "model": "gemini-2.5-flash"
}
```

**Response (SSE):**
```
data: {"searching": true}
data: {"searching": false}
data: {"generating": true}
data: {"content": "Based on the meeting notes..."}
data: {"content": " AQR discussed..."}
data: {"done": true}
```

### GET /api/chat/stats
Get document counts for selected filters.

### GET /api/chat/funds
Get list of all available fund names.

### GET /api/chat/portfolios
Get list of all portfolios.

---

## Environment Variables

```env
MONGO_URI=mongodb+srv://...
GEMINI_API_KEY=AIza...
```

---

## Summary

The chatbot implements a sophisticated RAG system that:

1. **Optimizes searches** with smart strategy detection (skip when possible)
2. **Combines semantic + keyword search** for comprehensive results
3. **Deduplicates factsheets** while showing relevant sections
4. **Streams responses** with real-time status updates
5. **Supports multiple models** including thinking variants
6. **Integrates web search** when additional context is needed
7. **Allows cancellation** with the Stop button

This architecture ensures fast, relevant, and comprehensive answers while minimizing unnecessary API calls and database queries.
