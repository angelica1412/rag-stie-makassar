# Struktur data yang digunakan pada backend

from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid

# ── Request dari pengguna ─────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

# ── Response ke pengguna ──────────────────────────────────────────────────────
class ChatResponse(BaseModel):
    status: str           # "found" | "not_found" | "hitl_pending"
    answer: Optional[str] = None
    sources: list[str] = []
    question_id: Optional[str] = None
    message: Optional[str] = None

# ── Pertanyaan yang menunggu jawaban HITL ─────────────────────────────────────
class PendingQuestion(BaseModel):
    question_id: str
    question: str
    user_id: Optional[str]
    timestamp: str
    status: str           # "pending" | "answered"
    answer: Optional[str] = None
    answered_at: Optional[str] = None

# ── Request jawaban dari admin QA ─────────────────────────────────────────────
class HITLAnswerRequest(BaseModel):
    question_id: str
    answer: str