# app/rag_pipeline.py
import re
import uuid
import numpy as np
from typing import TypedDict, List, Optional

import psycopg2
import psycopg2.extras

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from flashrank import Ranker, RerankRequest
from google import genai as genai_client_module
from langgraph.graph import StateGraph, END
from langsmith import traceable

pg_conn = None  # plain psycopg2 connection for chat_sessions/chat_messages

from app.config import (
    EMBEDDING_MODEL_NAME, QDRANT_URL, QDRANT_API_KEY,
    GEMINI_API_KEY, GEMINI_MODEL, MAX_MESSAGES_STM
)

embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
gemini_client = genai_client_module.Client(api_key=GEMINI_API_KEY)

all_chunks = {}
bm25_indexes = {}
chunk_vectors_cache = {}

# STM + LTM stores — set from main.py at startup
pg_store = None        # PostgresStore for LTM (user profile)
pg_checkpointer = None # PostgresSaver for STM (chat history)

USE_COMPRESSION = False


def get_chunk_vectors(subject):
    texts = [c["text"] for c in all_chunks[subject]]
    vectors = embed_model.encode(texts, normalize_embeddings=True, batch_size=32)
    return {c["chunk_id"]: v for c, v in zip(all_chunks[subject], vectors)}


def _summarize_search_io(inputs: dict) -> dict:
    return {"query": inputs.get("query"), "subject": inputs.get("subject"), "top_k": inputs.get("top_k")}


def _summarize_chunk_list_output(output):
    try:
        return {"count": len(output), "chunk_ids": [c.get("chunk_id") if isinstance(c, dict) else getattr(c, "id", None) for c in output][:10]}
    except Exception:
        return {"summary": "unavailable"}


def _summarize_candidates_io(inputs: dict) -> dict:
    candidates = inputs.get("candidates", [])
    return {"query": inputs.get("query"), "num_candidates": len(candidates) if candidates else 0, "top_k": inputs.get("top_k")}


def _summarize_pairs_output(output):
    try:
        return {"count": len(output), "chunk_ids": [c.get("chunk_id") for c, s in output][:10], "scores": [round(float(s), 4) for c, s in output][:10]}
    except Exception:
        return {"summary": "unavailable"}


@traceable(name="dense_search", process_inputs=_summarize_search_io, process_outputs=_summarize_chunk_list_output)
def dense_search(query, subject, top_k=10, book_filter=None):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_vector = embed_model.encode(query, normalize_embeddings=True).tolist()
    qdrant_filter = None
    if book_filter:
        qdrant_filter = Filter(must=[FieldCondition(key="book_name", match=MatchValue(value=book_filter))])
    return qdrant.query_points(collection_name=subject, query=query_vector, limit=top_k, query_filter=qdrant_filter).points


@traceable(name="sparse_search", process_inputs=_summarize_search_io)
def sparse_search(query, subject, top_k=10):
    tokenized_query = query.lower().split()
    scores = bm25_indexes[subject].get_scores(tokenized_query)
    top_indices = scores.argsort()[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


@traceable(name="rrf_fusion", process_outputs=_summarize_pairs_output)
def rrf_fusion(dense_results, sparse_results, chunks, k=60, top_k=5):
    scores = {}
    for rank, r in enumerate(dense_results):
        cid = r.payload["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    for rank, (idx, score) in enumerate(sparse_results):
        cid = chunks[idx]["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    id_to_chunk = {c["chunk_id"]: c for c in chunks}
    return [(id_to_chunk[cid], score) for cid, score in sorted_ids]


@traceable(name="hybrid_search", process_inputs=_summarize_search_io, process_outputs=_summarize_pairs_output)
def hybrid_search(query, subject, top_k=5, book_filter=None):
    dense = dense_search(query, subject, top_k=10, book_filter=book_filter)
    chunks = all_chunks[subject]
    if book_filter:
        chunks = [c for c in chunks if c.get("book_name") == book_filter]
    sparse = sparse_search(query, subject, top_k=10)
    return rrf_fusion(dense, sparse, chunks, top_k=top_k)


@traceable(name="mmr", process_outputs=_summarize_pairs_output)
def mmr(query_vector, candidates, chunk_vectors, top_k=5, lambda_param=0.5):
    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        best_score, best_item = -1, None
        for chunk, orig_score in remaining:
            vec = np.array(chunk_vectors[chunk["chunk_id"]])
            relevance = np.dot(query_vector, vec)
            sim_to_selected = max((np.dot(vec, np.array(chunk_vectors[s[0]["chunk_id"]])) for s in selected), default=0)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * sim_to_selected
            if mmr_score > best_score:
                best_score, best_item = mmr_score, (chunk, orig_score)
        selected.append(best_item)
        remaining.remove(best_item)
    return selected


@traceable(name="rerank", process_inputs=_summarize_candidates_io, process_outputs=_summarize_pairs_output)
def rerank(query, candidates, top_k=5):
    passages = [{"id": i, "text": chunk["text"]} for i, (chunk, score) in enumerate(candidates)]
    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    id_to_chunk = {i: chunk for i, (chunk, score) in enumerate(candidates)}
    return [(id_to_chunk[r["id"]], r["score"]) for r in reranked[:top_k]]


@traceable(name="long_context_reorder", process_outputs=_summarize_pairs_output)
def long_context_reorder(results):
    n = len(results)
    reordered = [None] * n
    left, right = 0, n - 1
    for i, item in enumerate(results):
        if i % 2 == 0:
            reordered[left] = item; left += 1
        else:
            reordered[right] = item; right -= 1
    return reordered


COMPRESS_PROMPT = """Given this question: "{query}"

Extract ONLY the sentences from the following text that are directly relevant to answering the question.
Keep exact original wording, don't paraphrase. If nothing is relevant, return "NOT_RELEVANT".

Text:
{text}

Relevant sentences:"""


@traceable(name="compress_chunk")
def compress_chunk(query, text):
    prompt = COMPRESS_PROMPT.format(query=query, text=text)
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=[prompt])
    result = response.text.strip().replace("Relevant sentences:", "").strip()
    return None if result == "NOT_RELEVANT" or not result else result


def contextual_compression(query, results):
    compressed = []
    for chunk, score in results:
        if chunk["content_type"] != "text":
            compressed.append((chunk, score, chunk["text"]))
            continue
        compressed_text = compress_chunk(query, chunk["text"])
        compressed.append((chunk, score, compressed_text or chunk["text"]))
    return compressed


# ─── LTM: User Profile ────────────────────────────────────────────────────────

MEMORY_EXTRACT_PROMPT = """You are responsible for maintaining a user profile for a pharmacy study assistant.

CURRENT USER PROFILE:
{user_profile}

USER'S LATEST MESSAGE:
{query}

TASK:
Extract any NEW facts worth remembering long-term about this user.
Only extract: name, study preferences, answer style preferences, subjects they focus on.
If the message contains nothing new or memory-worthy, respond with exactly: NONE
Otherwise respond with one fact per line, each as a short sentence.
Do NOT repeat what is already in the current profile."""


def load_user_profile(user_id: str) -> str:
    """Load user profile facts from PostgresStore."""
    if pg_store is None:
        return ""
    try:
        ns = ("user", user_id, "profile")
        items = pg_store.search(ns)
        return "\n".join(it.value.get("data", "") for it in items) if items else ""
    except Exception:
        return ""


def save_user_memory(user_id: str, query: str, current_profile: str):
    """Extract and save new user facts to PostgresStore."""
    if pg_store is None:
        return
    try:
        prompt = MEMORY_EXTRACT_PROMPT.format(user_profile=current_profile or "(empty)", query=query)
        response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=[prompt])
        result = response.text.strip()
        if result.upper() == "NONE" or not result:
            return
        ns = ("user", user_id, "profile")
        for line in result.splitlines():
            line = line.strip()
            if line:
                pg_store.put(ns, str(uuid.uuid4()), {"data": line})
    except Exception as e:
        print(f"Memory save error: {e}")


# ─── STM: Chat History ────────────────────────────────────────────────────────

def load_chat_history(thread_id: str) -> str:
    """Load recent chat history from chat_messages as formatted string."""
    if pg_conn is None:
        return ""
    try:
        with pg_conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM chat_messages WHERE thread_id = %s ORDER BY id DESC LIMIT %s",
                (thread_id, MAX_MESSAGES_STM),
            )
            rows = cur.fetchall()[::-1]  # back to chronological order
        history = [f"{'User' if role == 'human' else 'Assistant'}: {content[:500]}" for role, content in rows]
        return "\n".join(history)
    except Exception as e:
        print(f"Load history error: {e}")
        return ""



def save_chat_message(thread_id: str, subject: str, query: str, answer: str, user_id: str = "default_user"):
    if pg_conn is None:
        return
    try:
        with pg_conn.cursor() as cur:
            cur.execute(
                """INSERT INTO chat_sessions (thread_id, user_id, subject, title)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (thread_id) DO NOTHING""",
                (thread_id, user_id, subject, query[:60]),
            )
            cur.execute(
                "INSERT INTO chat_messages (thread_id, role, content) VALUES (%s, %s, %s)",
                (thread_id, "human", query),
            )
            cur.execute(
                "INSERT INTO chat_messages (thread_id, role, content) VALUES (%s, %s, %s)",
                (thread_id, "ai", answer),
            )
        pg_conn.commit()
    except Exception as e:
        print(f"Chat save error: {e}")
        pg_conn.rollback()

# ─── State ────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    subject: str
    intent: Optional[str]
    book_filter: Optional[str]
    retrieved_chunks: List[dict]
    context: str
    answer: str
    is_grounded: bool
    retry_count: int
    max_retries: int
    failure_reason: Optional[str]
    chat_history: Optional[str]   # STM
    user_profile: Optional[str]   # LTM


def _summarize_state_output(state: dict) -> dict:
    retrieved = state.get("retrieved_chunks", [])
    return {"query": state.get("query"), "subject": state.get("subject"), "num_retrieved": len(retrieved),
            "is_grounded": state.get("is_grounded"), "retry_count": state.get("retry_count")}


# ─── Intent ───────────────────────────────────────────────────────────────────

INTENT_PROMPT = """Classify this user message into exactly one category:

GREETING_SMALLTALK — greetings, thanks, casual chat, not a real question
GENERAL_KNOWLEDGE — a genuine question, but NOT about pharmacy/medical/science course content (e.g. asking about AI concepts, how this chatbot works, general definitions unrelated to the course subjects)
COURSE_QUESTION — a question that should be answered using the pharmacy/medical course materials (anatomy, biochemistry, physiology, organic chemistry, pharmacy)

Message: "{query}"

Respond with ONLY the category label, nothing else."""


@traceable(name="classify_intent_node")
def classify_intent_node(state: RAGState) -> RAGState:
    prompt = INTENT_PROMPT.format(query=state["query"])
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=[prompt])
    label = response.text.strip().upper()
    if label not in ("GREETING_SMALLTALK", "GENERAL_KNOWLEDGE", "COURSE_QUESTION"):
        label = "COURSE_QUESTION"
    state["intent"] = label
    return state


DIRECT_PROMPT = """You are a friendly pharmacy study assistant.

{user_profile_section}
{chat_history_section}

Respond naturally to this message. Keep it brief and conversational.

Message: {query}"""


@traceable(name="direct_answer_node")
def direct_answer_node(state: RAGState) -> RAGState:
    user_profile = state.get("user_profile") or ""
    chat_history = state.get("chat_history") or ""

    user_profile_section = f"User profile:\n{user_profile}" if user_profile else ""
    chat_history_section = f"Recent conversation:\n{chat_history}" if chat_history else ""

    prompt = DIRECT_PROMPT.format(
        user_profile_section=user_profile_section,
        chat_history_section=chat_history_section,
        query=state["query"]
    )
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=[prompt])
    state["answer"] = response.text.strip()
    state["is_grounded"] = True
    return state


def route_after_intent(state: RAGState) -> str:
    if state["intent"] == "COURSE_QUESTION":
        return "search"
    return "direct"


def classify_only(query: str) -> str:
    prompt = INTENT_PROMPT.format(query=query)
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=[prompt])
    label = response.text.strip().upper()
    if label not in ("GREETING_SMALLTALK", "GENERAL_KNOWLEDGE", "COURSE_QUESTION"):
        label = "COURSE_QUESTION"
    return label


# ─── Book Filter ──────────────────────────────────────────────────────────────

BOOK_EXTRACT_PROMPT = """The user is asking a question about "{subject}" course materials.
Available books in this subject: {book_list}

Does the user's message explicitly name one of these books? If yes, respond with the exact book name from the list.
If no specific book is named, respond with exactly: NONE

Message: "{query}"

Answer:"""


@traceable(name="extract_book_filter_node")
def extract_book_filter_node(state: RAGState) -> RAGState:
    subject = state["subject"]
    book_names = sorted({c.get("book_name") for c in all_chunks[subject] if c.get("book_name")})
    if not book_names:
        state["book_filter"] = None
        return state
    prompt = BOOK_EXTRACT_PROMPT.format(subject=subject, book_list=", ".join(book_names), query=state["query"])
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=[prompt])
    answer = response.text.strip()
    if answer.upper() == "NONE":
        state["book_filter"] = None
        return state
    match = next((b for b in book_names if b.lower() == answer.lower()), None)
    state["book_filter"] = match if match else "NOT_IN_DB"
    return state


# ─── Retrieve ─────────────────────────────────────────────────────────────────

@traceable(name="retrieve_node", process_outputs=_summarize_state_output)
def retrieve_node(state: RAGState) -> RAGState:
    if state.get("book_filter") == "NOT_IN_DB":
        state["answer"] = "Sorry, this book is not in my database yet. We will try to include it in later versions! 📚\n\nYou can ask without mentioning a specific book and I'll search all available books."
        state["retrieved_chunks"] = []
        state["is_grounded"] = True
        return state

    query, subject = state["query"], state["subject"]
    book_filter = state.get("book_filter")
    candidates = hybrid_search(query, subject, top_k=10, book_filter=book_filter)
    query_vec = embed_model.encode(query, normalize_embeddings=True)
    chunk_vecs = chunk_vectors_cache[subject]
    diverse = mmr(query_vec, candidates, chunk_vecs, top_k=8)
    reranked = rerank(query, diverse, top_k=5)
    reordered = long_context_reorder(reranked)

    if USE_COMPRESSION:
        compressed = contextual_compression(query, reordered)
        state["retrieved_chunks"] = [{"chunk": c, "score": s, "text_override": t} for c, s, t in compressed]
    else:
        state["retrieved_chunks"] = [{"chunk": c, "score": s} for c, s in reordered]
    return state


@traceable(name="build_context_node", process_outputs=lambda s: {"context_length": len(s.get("context", ""))})
def build_context_node(state: RAGState) -> RAGState:
    parts = []
    for item in state["retrieved_chunks"]:
        chunk = item["chunk"]
        book = chunk.get("book_name") or chunk["doc_id"]
        pages = chunk.get("pages")
        text = item.get("text_override", chunk["text"])
        parts.append(f"[Source: {book}, Page {pages}]\n{text}")
    state["context"] = "\n\n".join(parts)
    return state


# ─── Generation ───────────────────────────────────────────────────────────────

GENERATION_PROMPT = """You are an expert medical and pharmacy study assistant. Your job is to provide comprehensive, well-structured answers strictly based on the provided context.

{user_profile_section}
{chat_history_section}

## Instructions:
- Use ONLY the information from the context below. Do not use outside knowledge.
- Structure your answer clearly with headings, subheadings, and bullet points where appropriate.
- Be detailed and thorough — do not summarize too briefly.
- For every claim, cite the source in this format: (Source: book_name, Page X).
- If multiple sources support a point, cite all of them.
- If the context does not contain enough information, say: "I don't have enough information in the course materials to answer this."
- If the user says "explain this more" or "simplify this" — refer to the recent conversation above to understand what "this" refers to.

## Answer Format:
- Start with a brief 1-2 sentence introduction
- Use ## headings for major sections
- Use bullet points or numbered lists for steps, mechanisms, classifications, or item lists
- Use **bold** for key terms and important concepts
- End with a short summary if the answer is long

Context:
{context}

Question: {query}

Answer:"""


@traceable(name="generate_node")
def generate_node(state: RAGState) -> RAGState:
    user_profile = state.get("user_profile") or ""
    chat_history = state.get("chat_history") or ""

    user_profile_section = f"## User Profile:\n{user_profile}" if user_profile else ""
    chat_history_section = f"## Recent Conversation:\n{chat_history}" if chat_history else ""

    prompt = GENERATION_PROMPT.format(
        user_profile_section=user_profile_section,
        chat_history_section=chat_history_section,
        context=state["context"],
        query=state["query"]
    )
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=[prompt])
    answer = response.text.strip()

    citations = re.findall(r"\(Source:\s*(.*?),\s*Page\s*(.*?)\)", answer)
    seen = {}
    footnotes = []
    for book, page in citations:
        key = (book.strip(), page.strip())
        if key not in seen:
            seen[key] = len(seen) + 1
            footnotes.append(f"**[{seen[key]}]** {book.strip()}, Page {page.strip()}")

    def _replace(match):
        book, page = match.group(1).strip(), match.group(2).strip()
        return f"`[{seen[(book, page)]}]`"

    answer = re.sub(r"\(Source:\s*(.*?),\s*Page\s*(.*?)\)", _replace, answer)
    if footnotes:
        answer += "\n\n---\n**Sources:**\n" + "\n".join(footnotes)

    state["answer"] = answer
    return state


# ─── Grounding ────────────────────────────────────────────────────────────────

GROUNDING_CHECK_PROMPT = """Given this context and answer, determine if the answer is FULLY supported by the context, with no invented facts.

Context:
{context}

Answer:
{answer}

Respond in this exact format:
VERDICT: GROUNDED or NOT_GROUNDED
REASON: one short sentence explaining which claim (if any) lacks support. Write "N/A" if grounded."""


@traceable(name="check_grounding_node")
def check_grounding_node(state: RAGState) -> RAGState:
    prompt = GROUNDING_CHECK_PROMPT.format(context=state["context"], answer=state["answer"])
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=[prompt])
    text = response.text.strip()
    verdict_line = next((l for l in text.splitlines() if l.upper().startswith("VERDICT")), "VERDICT: NOT_GROUNDED")
    reason_line = next((l for l in text.splitlines() if l.upper().startswith("REASON")), "REASON: N/A")
    state["is_grounded"] = "NOT_GROUNDED" not in verdict_line.upper()
    state["failure_reason"] = reason_line.split(":", 1)[-1].strip() if not state["is_grounded"] else None
    return state


def route_after_check(state: RAGState) -> str:
    if state["is_grounded"]:
        return "end"
    if state["retry_count"] >= state["max_retries"]:
        return "end"
    return "retry"


def increment_retry_node(state: RAGState) -> RAGState:
    state["retry_count"] += 1
    if state.get("failure_reason"):
        state["query"] = f"{state['query']} (previous attempt failed: {state['failure_reason']}, focus retrieval on this gap)"
    return state


# ─── Graph ────────────────────────────────────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("extract_book_filter", extract_book_filter_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("generate", generate_node)
    graph.add_node("check_grounding", check_grounding_node)
    graph.add_node("increment_retry", increment_retry_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route_after_intent, {"search": "extract_book_filter", "direct": "direct_answer"})
    graph.add_edge("extract_book_filter", "retrieve")
    graph.add_edge("direct_answer", END)
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", "check_grounding")
    graph.add_conditional_edges("check_grounding", route_after_check, {"retry": "increment_retry", "end": END})
    graph.add_edge("increment_retry", "retrieve")
    return graph.compile()


rag_app = build_rag_graph()