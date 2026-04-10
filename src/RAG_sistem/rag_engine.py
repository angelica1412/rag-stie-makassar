from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "stie_documents"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"

SIMILARITY_THRESHOLD = 0.3

def load_index():

    # Set model
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)

    # Load ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    return index

def get_query_engine(index):
    """Buat query engine dari index dengan instruksi menjawab dalam Bahasa Indonesia."""

    # Custom prompt: paksa LLM menjawab dalam Bahasa Indonesia
    BAHASA_INDONESIA_PROMPT = PromptTemplate(
        "Kamu adalah asisten akademik STIE Ciputra Makassar yang menjawab "
        "pertanyaan berdasarkan dokumen internal kampus.\n"
        "Selalu jawab dalam Bahasa Indonesia yang baik dan benar.\n"
        "Jika informasi ada dalam konteks, jawab secara ringkas dan jelas.\n"
        "Jika informasi tidak ada dalam konteks, katakan bahwa informasi tidak tersedia "
        "dalam dokumen yang ada.\n\n"
        "Konteks dokumen:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Pertanyaan: {query_str}\n"
        "Jawaban (dalam Bahasa Indonesia):"
    )

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        streaming=False,
        text_qa_template=BAHASA_INDONESIA_PROMPT,
    )
    return query_engine

def query_documents(query_engine, question: str) -> dict:
    response = query_engine.query(question)
    source_nodes = response.source_nodes

    if not source_nodes:
        return {"status": "not_found", "answer": None, "sources": []}

    top_score = source_nodes[0].score if source_nodes[0].score else 0

    print(f"\n[DEBUG] Jumlah node ditemukan: {len(source_nodes)}")
    for i, node in enumerate(source_nodes):
        score = node.score if node.score else 0
        fname = node.metadata.get('file_name', 'Unknown')
        print(f"[DEBUG] Node {i+1}: skor={score:.4f}, file={fname}")
    print(f"[DEBUG] Top score: {top_score:.4f}, Threshold: {SIMILARITY_THRESHOLD}")

    if top_score < SIMILARITY_THRESHOLD:
        return {"status": "not_found", "answer": None, "sources": []}

    # Cek apakah LLM menyatakan informasi tidak tersedia
    answer_text = str(response)
    frasa_tidak_tersedia = [
        "tidak tersedia dalam dokumen",
        "tidak ada dalam dokumen",
        "tidak ditemukan dalam dokumen",
        "tidak ada informasi",
        "tidak disebutkan",
        "tidak terdapat",
        "maaf, informasi tersebut tidak tersedia",
    ]

    jawaban_tidak_ada = any(
        frasa in answer_text.lower()
        for frasa in frasa_tidak_tersedia
    )

    if jawaban_tidak_ada:
        print("[DEBUG] LLM menyatakan informasi tidak ada → HITL")
        return {"status": "not_found", "answer": None, "sources": []}

    # Kumpulkan sumber yang relevan
    sources = []
    for node in source_nodes:
        score = node.score if node.score else 0
        if score >= SIMILARITY_THRESHOLD:
            file_name = node.metadata.get("file_name", "Unknown")
            page = node.metadata.get("page_number", "?")
            source_info = f"{file_name} (hal. {page})"
            if source_info not in sources:
                sources.append(source_info)

    return {
        "status": "found",
        "answer": answer_text,
        "sources": sources
    }

if __name__ == "__main__":
    print("Memuat index dari ChromaDB...")
    index = load_index()
    query_engine = get_query_engine(index)
    print("Index berhasil dimuat!\n")

    # Test dengan pertanyaan
    while True:
        question = input("Masukkan pertanyaan (ketik 'keluar' untuk berhenti): ")
        if question.lower() == "keluar":
            break

        print("\nMencari jawaban...")
        result = query_documents(query_engine, question)

        if result["status"] == "found":
            print(f"\nJawaban: {result['answer']}")
            print(f"Sumber dokumen: {', '.join(result['sources'])}")
        elif result["status"] == "low_confidence":
            print(f"\nInformasi tidak ditemukan dengan keyakinan cukup (skor tertinggi: {result.get('top_score', 0):.4f}).")
            print("Pertanyaan ini akan diteruskan ke staf QA (HITL).")
        else:
            print("\nInformasi tidak ditemukan dalam dokumen (tidak ada node relevan).")
            print("Pertanyaan ini akan diteruskan ke staf QA (HITL).")
        print("\n" + "="*50 + "\n")