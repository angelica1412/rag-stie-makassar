import uuid
from datetime import datetime
from typing import Optional
from src.backend.models import PendingQuestion

# Simpan pertanyaan pending di memory
# Nanti bisa diganti database kalau perlu
pending_questions: dict[str, PendingQuestion] = {}

def add_pending_question(
    question: str,
    user_id: Optional[str] = None
) -> PendingQuestion:
    """Tambahkan pertanyaan baru yang perlu dijawab admin QA."""
    question_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    pending = PendingQuestion(
        question_id=question_id,
        question=question,
        user_id=user_id,
        timestamp=timestamp,
        status="pending",
        answer=None,
        answered_at=None
    )
    pending_questions[question_id] = pending
    print(f"[HITL] Pertanyaan baru masuk: {question_id} — {question[:50]}...")
    return pending

def get_all_pending() -> list[PendingQuestion]:
    """Ambil semua pertanyaan yang belum dijawab."""
    return [
        q for q in pending_questions.values()
        if q.status == "pending"
    ]

def get_all_questions() -> list[PendingQuestion]:
    """Ambil semua pertanyaan termasuk yang sudah dijawab."""
    return list(pending_questions.values())

def answer_question(
    question_id: str,
    answer: str
) -> Optional[PendingQuestion]:
    """Admin QA menjawab pertanyaan."""
    if question_id not in pending_questions:
        return None

    question = pending_questions[question_id]
    question.status = "answered"
    question.answer = answer
    question.answered_at = datetime.now().isoformat()

    print(f"[HITL] Pertanyaan {question_id} sudah dijawab.")
    return question

def get_question_by_id(question_id: str) -> Optional[PendingQuestion]:
    """Ambil pertanyaan berdasarkan ID."""
    return pending_questions.get(question_id)