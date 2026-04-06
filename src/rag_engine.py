from llama_index.core import VectorStoreIndex, StorageContext, Settings
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
    """Load index dari ChromaDB yang sudah ada."""

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
    """Buat query engine dari index."""
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        streaming=False,
    )
    return query_engine

def query_documents(query_engine, question: str) -> dict:
    """
    Kirim pertanyaan ke sistem RAG.
    Return dict berisi jawaban dan status (ditemukan atau perlu HITL).
    """
    response = query_engine.query(question)

    # Cek apakah ada dokumen yang relevan ditemukan
    source_nodes = response.source_nodes

    if not source_nodes:
        return {
            "status": "not_found",
            "answer": None,
            "sources": []
        }

    # Cek skor relevansi tertinggi
    top_score = source_nodes[0].score if source_nodes[0].score else 0

    # Debug: tampilkan skor semua node yang ditemukan
    print(f"\n[DEBUG] Jumlah node ditemukan: {len(source_nodes)}")
    for i, node in enumerate(source_nodes):
        score = node.score if node.score else 0
        fname = node.metadata.get('file_name', 'Unknown')
        print(f"[DEBUG] Node {i+1}: skor={score:.4f}, file={fname}")
    print(f"[DEBUG] Top score: {top_score:.4f}, Threshold: {SIMILARITY_THRESHOLD}")

    if top_score < SIMILARITY_THRESHOLD:
        return {
            "status": "low_confidence",
            "answer": None,
            "sources": [],
            "top_score": top_score
        }

    # Kumpulkan referensi dokumen sumber
    sources = []
    for node in source_nodes:
        file_name = node.metadata.get("file_name", "Unknown")
        if file_name not in sources:
            sources.append(file_name)

    return {
        "status": "found",
        "answer": str(response),
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