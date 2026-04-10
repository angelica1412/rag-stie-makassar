from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.backend.models import ChatRequest, ChatResponse, HITLAnswerRequest
from src.backend.hitl_manager import (
    add_pending_question,
    get_all_pending,
    get_all_questions,
    answer_question,
    get_question_by_id
)
from src.RAG_sistem.rag_engine import load_index, get_query_engine, query_documents

# ── Simpan index dan query engine di memory ───────────────────────────────────
rag_index = None
rag_query_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG index saat server pertama kali dijalankan."""
    global rag_index, rag_query_engine
    print("Memuat RAG index...")
    rag_index = load_index()
    rag_query_engine = get_query_engine(rag_index)
    print("RAG index siap!")
    yield
    print("Server dimatikan.")

# ── Inisialisasi FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="Sistem Tanya Jawab STIE Ciputra Makassar",
    description="RAG-based QA system dengan mekanisme HITL",
    version="1.0.0",
    lifespan=lifespan
)

# Izinkan akses dari React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT UNTUK SIVITAS AKADEMIKA
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "Sistem Tanya Jawab STIE Ciputra Makassar API"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint utama untuk menerima pertanyaan dari sivitas akademika.
    Kalau tidak ditemukan → otomatis masuk HITL queue.
    """
    if not rag_query_engine:
        raise HTTPException(
            status_code=503,
            detail="RAG engine belum siap"
        )

    # Cari jawaban di dokumen
    result = query_documents(rag_query_engine, request.question)

    if result["status"] == "found":
        return ChatResponse(
            status="found",
            answer=result["answer"],
            sources=result["sources"],
            message="Jawaban ditemukan dari dokumen internal."
        )
    else:
        # Tidak ditemukan → masuk HITL queue
        pending = add_pending_question(
            question=request.question,
            user_id=request.user_id
        )
        return ChatResponse(
            status="hitl_pending",
            answer=None,
            sources=[],
            question_id=pending.question_id,
            message="Pertanyaan kamu sedang diproses oleh staf QA. "
                   "Estimasi waktu tunggu 1-3 menit."
        )

@app.get("/chat/status/{question_id}")
def check_status(question_id: str):
    """
    Pengguna bisa cek apakah pertanyaannya sudah dijawab admin.
    """
    question = get_question_by_id(question_id)
    if not question:
        raise HTTPException(
            status_code=404,
            detail="Pertanyaan tidak ditemukan"
        )
    return question

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT UNTUK ADMIN QA (HITL DASHBOARD)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/admin/questions")
def get_pending_questions():
    """Ambil semua pertanyaan yang belum dijawab untuk dashboard admin."""
    return get_all_pending()

@app.get("/admin/questions/all")
def get_all_questions_endpoint():
    """Ambil semua pertanyaan termasuk yang sudah dijawab."""
    return get_all_questions()

@app.post("/admin/answer")
def submit_answer(request: HITLAnswerRequest):
    """
    Admin QA menjawab pertanyaan yang masuk ke HITL queue.
    """
    question = answer_question(
        question_id=request.question_id,
        answer=request.answer
    )
    if not question:
        raise HTTPException(
            status_code=404,
            detail="Pertanyaan tidak ditemukan"
        )
    return {
        "message": "Jawaban berhasil disimpan",
        "question": question
    }