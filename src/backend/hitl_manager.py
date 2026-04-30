import uuid
from datetime import datetime
from typing import Optional
from src.backend.models import PendingQuestion

pending_questions: dict[str, PendingQuestion] = {}

def add_pending_question(
    question: str,
    user_id: Optional[str] = None
) -> PendingQuestion:
    """Tambahkan pertanyaan baru ke antrian HITL."""
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
    print(f"[HITL] Pertanyaan baru: {question_id} — {question[:50]}...")
    return pending


def get_all_pending() -> list[PendingQuestion]:
    return [
        q for q in pending_questions.values()
        if q.status == "pending"
    ]


def get_all_questions() -> list[PendingQuestion]:
    return list(pending_questions.values())


def answer_question(
    question_id: str,
    answer: str
) -> Optional[PendingQuestion]:
    """Admin QA menjawab pertanyaan dan simpan ke ChromaDB."""
    if question_id not in pending_questions:
        return None

    question = pending_questions[question_id]
    question.status = "answered"
    question.answer = answer
    question.answered_at = datetime.now().isoformat()

    print(f"[HITL] Pertanyaan {question_id} dijawab.")

    # Simpan ke ChromaDB
    try:
        _save_to_knowledge_base(question.question, answer)
        print(f"[HITL] Jawaban berhasil disimpan ke basis pengetahuan.")
    except Exception as e:
        print(f"[HITL] Gagal menyimpan ke basis pengetahuan: {e}")

    return question


def _save_to_knowledge_base(question: str, answer: str):
    """
    Simpan pasangan pertanyaan-jawaban HITL ke ChromaDB
    supaya bisa dijawab otomatis di masa mendatang.
    """
    import chromadb
    from llama_index.core import VectorStoreIndex, StorageContext, Settings
    from llama_index.core import Document
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding

    CHROMA_PATH = "./data/chroma_db"
    COLLECTION_NAME = "stie_documents"
    EMBED_MODEL = "qwen3-embedding"

    # Setup embedding
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None

    # Buat dokumen dari pasangan Q&A
    doc_content = (
        f"Pertanyaan: {question}\n"
        f"Jawaban: {answer}\n"
        f"Sumber: Jawaban manual dari staf QA STIE Ciputra Makassar"
    )

    doc = Document(
        text=doc_content,
        metadata={
            "file_name": "HITL_Knowledge_Base",
            "source": "HITL",
            "tipe_dokumen": "naratif",
            "chunk_type": "hitl",
            "page_number": 1,
            "question": question,
            "answered_at": datetime.now().isoformat()
        }
    )

    # Simpan ke ChromaDB yang sudah ada
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(
        COLLECTION_NAME
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # Tambahkan ke index yang sudah ada
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    index.insert(doc)

    print(f"[HITL] Dokumen berhasil ditambahkan ke ChromaDB.")


def get_question_by_id(
    question_id: str
) -> Optional[PendingQuestion]:
    return pending_questions.get(question_id)