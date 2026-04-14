from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import chromadb
import ollama as ollama_client

CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "stie_documents"
EMBED_MODEL = "qwen3-embedding"
LLM_MODEL = "qwen2.5:7b"
SIMILARITY_THRESHOLD = 0.3

# Token awal yang menandakan LLM tidak tahu jawabannya
TIDAK_TAHU_TOKENS = [
    "maaf",
    "tidak tersedia",
    "tidak ditemukan",
    "tidak ada",
    "tidak disebutkan",
    "tidak terdapat",
    "informasi tersebut tidak",
    "tidak memiliki informasi",
    "tidak dapat menemukan",
    "i cannot",
    "i don't",
    "i do not",
    "no information",
    "not found",
    "not available",
]

def load_index():
    """Load index dari ChromaDB yang sudah ada."""
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    return index

def get_query_engine(index):
    """Buat query engine untuk retrieval saja — tanpa generate jawaban."""
    qa_prompt = PromptTemplate(
        "Kamu adalah asisten sistem tanya jawab dokumen internal "
        "STIE Ciputra Makassar. Jawab pertanyaan HANYA berdasarkan "
        "informasi yang ada dalam konteks dokumen berikut.\n\n"
        "Konteks dokumen:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Aturan penting:\n"
        "1. Jawab SELALU dalam Bahasa Indonesia\n"
        "2. Jika informasi tidak ada dalam konteks, jawab PERSIS dengan: "
        "'Maaf, informasi tersebut tidak tersedia dalam dokumen internal kampus.'\n"
        "3. Jangan mengarang jawaban\n"
        "4. Jangan menyarankan untuk mencari di tempat lain\n\n"
        "Pertanyaan: {query_str}\n"
        "Jawaban: "
    )

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        streaming=False,
        text_qa_template=qa_prompt
    )
    return query_engine

def retrieve_context(index, question: str) -> tuple[list, str]:
    """
    Ambil konteks dokumen yang relevan dari ChromaDB.
    Return: (source_nodes, context_text)
    """
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(question)
    
    # Gabungkan teks dari semua node menjadi satu konteks
    context_parts = []
    for node in nodes:
        context_parts.append(node.get_content())
    
    context_text = "\n\n---\n\n".join(context_parts)
    return nodes, context_text

def stream_with_interrupt(question: str, context: str) -> tuple[str, bool]:
    """
    Generate jawaban menggunakan Ollama streaming.
    Pantau token awal — kalau menunjukkan tidak tahu, hentikan streaming.
    
    Return: (answer_text, is_found)
    - is_found=True  → jawaban ditemukan, tampilkan ke pengguna
    - is_found=False → tidak ditemukan, lempar ke HITL
    """
    prompt = (
        f"Kamu adalah asisten sistem tanya jawab dokumen internal "
        f"STIE Ciputra Makassar. Jawab pertanyaan HANYA berdasarkan "
        f"informasi yang ada dalam konteks dokumen berikut.\n\n"
        f"Konteks dokumen:\n"
        f"---------------------\n"
        f"{context}\n"
        f"---------------------\n\n"
        f"Aturan penting:\n"
        f"1. Jawab SELALU dalam Bahasa Indonesia\n"
        f"2. Jika informasi tidak ada dalam konteks, mulai jawaban dengan: "
        f"'Maaf, informasi tersebut tidak tersedia'\n"
        f"3. Jangan mengarang jawaban\n\n"
        f"Pertanyaan: {question}\n"
        f"Jawaban: "
    )

    full_answer = ""
    checked = False 
    is_found = True

    print("[STREAMING] Mulai generate jawaban...")

    try:
        # Gunakan Ollama streaming langsung
        stream = ollama_client.generate(
            model=LLM_MODEL,
            prompt=prompt,
            stream=True
        )

        for chunk in stream:
            token = chunk.get("response", "")
            full_answer += token

            # Cek 50 karakter pertama untuk deteksi dini
            if not checked and len(full_answer) >= 50:
                checked = True
                answer_lower = full_answer.lower().strip()

                tidak_tahu = any(
                    phrase in answer_lower
                    for phrase in TIDAK_TAHU_TOKENS
                )

                if tidak_tahu:
                    print(f"[STREAMING] Deteksi dini: LLM tidak tahu → stop streaming")
                    print(f"[STREAMING] Token awal: '{full_answer[:80]}'")
                    is_found = False
                    break

            if checked and is_found:
                print(token, end="", flush=True)

            if chunk.get("done", False):
                break

    except Exception as e:
        print(f"[STREAMING] Error: {e}")
        is_found = False

    print() 
    return full_answer, is_found


def query_documents(query_engine, question: str, index=None) -> dict:
    """
    Kirim pertanyaan ke sistem RAG dengan interruptible streaming.
    """
    # Ambil konteks dari ChromaDB
    if index is None:
        # Fallback ke cara lama kalau index tidak diberikan
        response = query_engine.query(question)
        source_nodes = response.source_nodes
    else:
        source_nodes, context_text = retrieve_context(index, question)

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
        print("[DEBUG] Skor di bawah threshold → HITL")
        return {"status": "not_found", "answer": None, "sources": []}

    # Generate jawaban dengan interruptible streaming
    if index is not None:
        answer, is_found = stream_with_interrupt(question, context_text)
    else:
        answer = str(response)
        is_found = True

    if not is_found:
        print("[DEBUG] Streaming dihentikan — jawaban tidak ada → HITL")
        return {"status": "not_found", "answer": None, "sources": []}

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
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    print("Memuat index dari ChromaDB...")
    index = load_index()
    query_engine = get_query_engine(index)
    print("Index berhasil dimuat!\n")

    while True:
        question = input("Masukkan pertanyaan (ketik 'keluar' untuk berhenti): ")
        if question.lower() == "keluar":
            break

        print("\nMencari jawaban...")
        # Kirim index supaya bisa pakai interruptible streaming
        result = query_documents(query_engine, question, index=index)

        if result["status"] == "found":
            print(f"\nJawaban: {result['answer']}")
            print(f"Sumber dokumen: {', '.join(result['sources'])}")
        else:
            print("\nInformasi tidak ditemukan dalam dokumen.")
            print("Pertanyaan ini akan diteruskan ke staf QA (HITL).")
        print("\n" + "="*50 + "\n")