# app/main.py
import psycopg2
import uuid
from fastapi import FastAPI, HTTPException, Header
import jwt
import os
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import numpy as np
from huggingface_hub import hf_hub_download
import shutil
from fastapi.responses import FileResponse
from datetime import datetime, timezone

from app.config import (
    SUBJECTS, IMAGES_DIR, FORMULAS_DIR, CHUNK_VECTORS_DIR,
    HF_TOKEN, HF_REPO, POSTGRES_URI
)
from app.data_loader import load_all_chunks, build_bm25_indexes, ensure_table_image
from app.models import QueryRequest, QueryResponse, SourceItem, ChatSession, ChatHistoryResponse
import app.rag_pipeline as rag_pipeline
from qdrant_client.models import PayloadSchemaType

from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver


SUPABASE_PROJECT_URL = os.environ.get("SUPABASE_PROJECT_URL")
_jwks_client = jwt.PyJWKClient(f"{SUPABASE_PROJECT_URL}/auth/v1/.well-known/jwks.json")

def is_valid_uuid(val: str) -> bool:
    if not val:
        return False
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, TypeError, AttributeError):
        return False

def get_current_user_id(authorization: str = Header(None), default_user: str = "default_user") -> str:
    if not authorization:
        return default_user
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid auth token")
    token = authorization.split(" ")[1]
    if not token or token in ("undefined", "null", "none"):
        return default_user
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token, signing_key.key, algorithms=["ES256"], audience="authenticated"
        )
        return payload["sub"]
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

def _load_chunk_vectors(subject, chunks):
    local_path = CHUNK_VECTORS_DIR / f"{subject}_vectors.npz"
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"chunk_vectors_cache/{subject}_vectors.npz",
            repo_type="dataset",
            token=HF_TOKEN
        )
        shutil.copy(downloaded, local_path)
    data = np.load(local_path, allow_pickle=True)
    return {cid: vec for cid, vec in zip(data["ids"], data["vectors"])}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Chunks + BM25 ──────────────────────────────────────────
    print("Loading chunks and BM25 indexes...")
    all_chunks = load_all_chunks()
    bm25_indexes = build_bm25_indexes(all_chunks)
    rag_pipeline.all_chunks = all_chunks
    rag_pipeline.bm25_indexes = bm25_indexes

    # ── Qdrant payload indexes ─────────────────────────────────
    print("Creating Qdrant payload indexes...")
    for subject in SUBJECTS:
        try:
            rag_pipeline.qdrant.create_payload_index(
                collection_name=subject,
                field_name="book_name",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass  # already exists

    # ── Chunk vectors ──────────────────────────────────────────
    print("Loading chunk vectors...")
    rag_pipeline.chunk_vectors_cache = {
        subject: _load_chunk_vectors(subject, all_chunks[subject]) for subject in SUBJECTS
    }

    # ── Postgres STM + LTM ─────────────────────────────────────
    print("Setting up Postgres STM + LTM...")
    # Setup tables once
    with PostgresSaver.from_conn_string(POSTGRES_URI) as saver:
        saver.setup()

    # Persistent connections
    pg_store = PostgresStore.from_conn_string(POSTGRES_URI)
    rag_pipeline.pg_store = pg_store.__enter__()

    rag_pipeline.pg_conn = psycopg2.connect(POSTGRES_URI)
    rag_pipeline.pg_conn.autocommit = True

    print("Startup complete.")
    yield

    # ── Cleanup ────────────────────────────────────────────────
    pg_store.__exit__(None, None, None)
    rag_pipeline.pg_conn.close()


app = FastAPI(title="Pharmacy RAG API", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
cors_origins = [
    "http://localhost:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5173",
    "null",
]
custom_frontend = os.environ.get("FRONTEND_URL")
if custom_frontend:
    cors_origins.append(custom_frontend)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FORMULAS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
app.mount("/formulas", StaticFiles(directory=str(FORMULAS_DIR)), name="formulas")


@app.get("/tables/{filename}")
def get_table_image(filename: str):
    path = ensure_table_image(filename)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Table image not found")
    return FileResponse(path)


@app.get("/")
def root():
    return {"status": "ok", "subjects": SUBJECTS}


@app.get("/books/{subject}")
def get_books(subject: str):
    if subject not in SUBJECTS:
        raise HTTPException(status_code=400, detail="Invalid subject.")
    books = sorted({c.get("book_name") for c in rag_pipeline.all_chunks[subject] if c.get("book_name")})
    return {"subject": subject, "books": books}


@app.post("/classify")
def classify(request: QueryRequest):
    intent = rag_pipeline.classify_only(request.query)
    return {"intent": intent}


@app.get("/chats/{user_id}", response_model=ChatHistoryResponse)
def get_chat_sessions(user_id: str, authorization: str = Header(None)):
    user_id = get_current_user_id(authorization, default_user=user_id)
    if rag_pipeline.pg_conn is None or not is_valid_uuid(user_id):
        return ChatHistoryResponse(sessions=[])
    try:
        with rag_pipeline.pg_conn.cursor() as cur:
            cur.execute(
                """SELECT s.thread_id, s.subject, s.title, s.created_at,
                          (SELECT content FROM chat_messages m
                           WHERE m.thread_id = s.thread_id AND m.role = 'ai'
                           ORDER BY m.id DESC LIMIT 1) AS last_message
                   FROM chat_sessions s
                   LEFT JOIN (
                       SELECT thread_id, MAX(timestamp) AS last_activity
                       FROM chat_messages GROUP BY thread_id
                   ) m ON s.thread_id = m.thread_id
                   WHERE s.user_id = %s
                   ORDER BY COALESCE(m.last_activity, s.created_at) DESC""",
                (user_id,),
            )
            rows = cur.fetchall()
        sessions = [
            ChatSession(
                thread_id=r[0],
                subject=r[1],
                title=r[2],
                created_at=r[3].isoformat() if r[3] else "",
                last_message=(r[4][:80] if r[4] else ""),
            )
            for r in rows
        ]
        return ChatHistoryResponse(sessions=sessions)
    except Exception as e:
        print(f"Chat sessions error: {e}")
        return ChatHistoryResponse(sessions=[])
 

@app.get("/chats/{user_id}/{thread_id}/messages")
def get_chat_messages(user_id: str, thread_id: str, authorization: str = Header(None)):
    user_id = get_current_user_id(authorization, default_user=user_id)
    if rag_pipeline.pg_conn is None:
        return {"messages": []}
    try:
        with rag_pipeline.pg_conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, timestamp FROM chat_messages WHERE thread_id = %s ORDER BY id",
                (thread_id,),
            )
            rows = cur.fetchall()
        return {"messages": [
            {"role": "user" if role == "human" else "assistant", "content": content, "timestamp": ts.isoformat()}
            for role, content, ts in rows
        ]}
    except Exception as e:
        print(f"Get messages error: {e}")
        return {"messages": []}


@app.delete("/chats/{user_id}/{thread_id}")
def delete_chat(user_id: str, thread_id: str, authorization: str = Header(None)):
    user_id = get_current_user_id(authorization, default_user=user_id)
    if rag_pipeline.pg_conn is None:
        return {"status": "error"}
    try:
        with rag_pipeline.pg_conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE thread_id = %s", (thread_id,))
        rag_pipeline.pg_conn.commit()
        return {"status": "deleted"}
    except Exception as e:
        rag_pipeline.pg_conn.rollback()
        return {"status": "error", "detail": str(e)}


@app.get("/dashboard/stats")
def dashboard_stats():
    if rag_pipeline.pg_conn is None:
        return {"error": "No database connection"}
    try:
        with rag_pipeline.pg_conn.cursor() as cur:
            # Top-line totals
            cur.execute("SELECT COUNT(DISTINCT user_id) FROM chat_sessions")
            total_users = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chat_sessions")
            total_threads = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chat_messages")
            total_messages = cur.fetchone()[0]

            # Per-user breakdown
            cur.execute("""
                SELECT s.user_id, COUNT(DISTINCT s.thread_id) AS threads, COUNT(m.id) AS messages
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON s.thread_id = m.thread_id
                GROUP BY s.user_id
                ORDER BY messages DESC
            """)
            per_user = [{"user_id": r[0], "threads": r[1], "messages": r[2]} for r in cur.fetchall()]

            # Top subjects
            cur.execute("""
                SELECT subject, COUNT(*) AS thread_count
                FROM chat_sessions
                GROUP BY subject
                ORDER BY thread_count DESC
            """)
            top_subjects = [{"subject": r[0], "count": r[1]} for r in cur.fetchall()]

            # Messages per day (last 30 days)
            cur.execute("""
                SELECT DATE(timestamp) AS day, COUNT(*) AS count
                FROM chat_messages
                WHERE timestamp > now() - interval '30 days'
                GROUP BY day
                ORDER BY day
            """)
            messages_over_time = [{"day": r[0].isoformat(), "count": r[1]} for r in cur.fetchall()]

            # Recent activity feed
            cur.execute("""
                SELECT s.user_id, s.subject, s.title, m.role, m.content, m.timestamp
                FROM chat_messages m
                JOIN chat_sessions s ON m.thread_id = s.thread_id
                ORDER BY m.timestamp DESC
                LIMIT 15
            """)
            recent_activity = [
                {"user_id": r[0], "subject": r[1], "title": r[2], "role": r[3], "content": r[4][:100], "timestamp": r[5].isoformat()}
                for r in cur.fetchall()
            ]

            # DB size per table
            cur.execute("""
                SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
                FROM pg_catalog.pg_statio_user_tables
                ORDER BY pg_total_relation_size(relid) DESC
            """)
            table_sizes = [{"table": r[0], "size": r[1]} for r in cur.fetchall()]

        return {
            "totals": {"users": total_users, "threads": total_threads, "messages": total_messages},
            "per_user": per_user,
            "top_subjects": top_subjects,
            "messages_over_time": messages_over_time,
            "recent_activity": recent_activity,
            "table_sizes": table_sizes,
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, authorization: str = Header(None)):
    if request.subject not in SUBJECTS:
        raise HTTPException(status_code=400, detail=f"Invalid subject. Choose from: {SUBJECTS}")

    # Generate thread_id if new chat
    thread_id = request.thread_id or str(uuid.uuid4())
    user_id = get_current_user_id(authorization, default_user=request.user_id or "default_user")

    # Load STM + LTM
    chat_history = rag_pipeline.load_chat_history(thread_id)
    user_profile = rag_pipeline.load_user_profile(user_id)

    initial_state = {
        "query": request.query,
        "subject": request.subject,
        "intent": None,
        "book_filter": None,
        "retrieved_chunks": [],
        "context": "",
        "answer": "",
        "is_grounded": False,
        "retry_count": 0,
        "max_retries": request.max_retries,
        "failure_reason": None,
        "chat_history": chat_history,
        "user_profile": user_profile,
    }

    result = rag_pipeline.rag_app.invoke(initial_state)

    # Save to STM
    rag_pipeline.save_chat_message(
        thread_id=thread_id,
        subject=request.subject,
        query=request.query,
        answer=result["answer"],
        user_id=user_id,
    )

    # Save to LTM (only for non-RAG turns to save tokens)
    if result.get("intent") in ("GREETING_SMALLTALK", "GENERAL_KNOWLEDGE"):
        rag_pipeline.save_user_memory(user_id, request.query, user_profile)

    sources = []
    for item in result["retrieved_chunks"]:
        chunk = item["chunk"]
        sources.append(SourceItem(
            doc_id=chunk["doc_id"],
            book_name=chunk.get("book_name"),
            content_type=chunk["content_type"],
            pages=chunk.get("pages", []),
            text=chunk["text"],
            image_path=chunk.get("image_path"),
            formula_image_path=chunk.get("formula_image_path"),
            table_image_path=chunk.get("table_image_path"),
            score=item["score"],
        ))

    return QueryResponse(
        answer=result["answer"],
        is_grounded=result["is_grounded"],
        retry_count=result["retry_count"],
        intent=result.get("intent"),
        sources=sources,
        thread_id=thread_id,
    )