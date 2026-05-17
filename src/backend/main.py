from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from src.backend.models import ChatRequest, ChatResponse, HITLAnswerRequest
from src.backend.hitl_manager import (
    add_pending_question,
    get_all_pending,
    get_all_questions,
    answer_question,
    get_question_by_id
)
from src.RAG_sistem.rag_engine import load_index, get_query_engine, query_documents

import os

FORM_PATH = "./data/documents/form"

# Simpan index dan query engine di memory
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

# Inisialisasi FastAPI 
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
    result = query_documents(
        rag_query_engine,
        request.question,
        index=rag_index
    )

    if result["status"] == "found":
        # Konversi sources form ke string kalau perlu
        sources = result.get("sources", [])
        is_form_response = result.get("is_form_response", False)

        return ChatResponse(
            status="found",
            answer=result["answer"],
            sources=sources, 
            message="Jawaban ditemukan dari dokumen internal.",
            is_form_response=is_form_response
        )
    elif result["status"] == "not_relevant":
        return ChatResponse(
            status="not_relevant",
            answer=result["answer"],
            sources=[],
            message=result["answer"]
        )
    else:
        pending = add_pending_question(
            question=request.question,
            user_id=request.user_id
        )
        return ChatResponse(
            status="hitl_pending",
            answer=None,
            sources=[],
            question_id=pending.question_id,
            message=result.get("message") or (
                "Maaf, pertanyaan Anda tidak ditemukan dalam dokumen "
                "internal yang tersedia pada sistem ini. Pertanyaan Anda "
                "akan diteruskan kepada staf QA STIE Ciputra Makassar "
                "untuk mendapatkan jawaban yang tepat. "
                "Mohon tunggu beberapa saat."
            )
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

@app.get("/preview/{filename}")
def preview_form(filename: str):
    """Preview file form di browser."""
    filepath = os.path.join(FORM_PATH, filename)

    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail="File tidak ditemukan"
        )

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        # PDF bisa langsung ditampilkan di browser
        return FileResponse(
            path=filepath,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={filename}"
            }
        )
    else:
        # DOCX/XLSX tidak bisa preview — redirect ke download
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

@app.get("/download/{filename}")
def download_form(filename: str):
    """Download file form."""
    filepath = os.path.join(FORM_PATH, filename)

    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail="File tidak ditemukan"
        )

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

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