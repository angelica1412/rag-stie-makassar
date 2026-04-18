from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator
)
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

FORM_KEYWORDS = [
    "form", "formulir", "template", "dokumen apa",
    "format", "blanko", "unduh", "download",
    "form apa", "formulir apa", "menggunakan form",
    "form yang digunakan", "formulir yang digunakan",
    "form untuk", "formulir untuk"
]

def detect_query_type(question: str) -> str:
    """
    Deteksi apakah pertanyaan tentang form atau naratif.
    Return: 'form' atau 'naratif'
    """
    question_lower = question.lower()
    is_form_query = any(
        keyword in question_lower
        for keyword in FORM_KEYWORDS
    )
    return "form" if is_form_query else "naratif"

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
    """Buat query engine dari index."""
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
        "2. Jika pertanyaan meminta prosedur atau langkah-langkah, "
        "sebutkan SEMUA langkah secara lengkap tanpa ada yang terlewat\n"
        "3. Jika informasi tidak ada dalam konteks, jawab PERSIS dengan: "
        "'Maaf, informasi tersebut tidak tersedia dalam dokumen internal kampus.'\n"
        "4. Jangan mengarang jawaban\n"
        "5. Jangan menyarankan untuk mencari di tempat lain\n"
        "6. Jika ada nomor form, kode dokumen, atau istilah teknis "
        "yang disebutkan dalam konteks, sertakan dalam jawaban\n"
        "7. Jawab secara terstruktur dan lengkap\n\n"
        "Pertanyaan: {query_str}\n"
        "Jawaban: "
    )

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        streaming=False,
        text_qa_template=qa_prompt
    )
    return query_engine

def retrieve_context(index, question: str, query_type: str) -> tuple[list, str]:
    """
    Ambil konteks dokumen berdasarkan tipe query.
    - query_type='naratif' → hanya cari di dokumen naratif
    - query_type='form'    → hanya cari di dokumen form
    """
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tipe_dokumen",
                value=query_type,
                operator=FilterOperator.EQ
            )
        ]
    )

    retriever = index.as_retriever(
        similarity_top_k=5,
        filters=filters
    )

    nodes = retriever.retrieve(question)

    context_parts = []
    for node in nodes:
        context_parts.append(node.get_content())

    context_text = "\n\n---\n\n".join(context_parts)
    return nodes, context_text

def enhance_query(question: str) -> str:
    """
    Perkaya pertanyaan pengguna secara otomatis
    sebelum dilakukan pencarian semantik.
    """
    enhance_prompt = (
        f"Kamu adalah asisten yang membantu memperjelas pertanyaan. "
        f"Tugas kamu adalah menulis ulang pertanyaan berikut menjadi "
        f"lebih spesifik dan lengkap dalam Bahasa Indonesia, "
        f"dengan menambahkan konteks bahwa pertanyaan ini berkaitan "
        f"dengan dokumen internal STIE Ciputra Makassar. "
        f"Jangan tambahkan informasi baru, hanya perjelas pertanyaannya. "
        f"Berikan HANYA pertanyaan yang diperjelas, tanpa penjelasan tambahan.\n\n"
        f"Pertanyaan asli: {question}\n"
        f"Pertanyaan yang diperjelas: "
    )

    try:
        response = ollama_client.generate(
            model=LLM_MODEL,
            prompt=enhance_prompt,
            stream=False,
            options={"temperature": 0.1}
        )
        enhanced = response.get("response", "").strip()
        print(f"[DEBUG] Query asli    : {question}")
        print(f"[DEBUG] Query enhanced: {enhanced}")
        return enhanced if enhanced else question
    except Exception:
        return question

def stream_with_interrupt(question: str, context: str) -> tuple[str, bool]:
    """
    Generate jawaban menggunakan Ollama streaming.
    Pantau token awal — kalau menunjukkan tidak tahu, hentikan streaming.
    """
    prompt = (
    f"Kamu adalah asisten sistem tanya jawab dokumen internal "
    f"STIE Ciputra Makassar. Jawab pertanyaan HANYA berdasarkan "
    f"informasi yang ada dalam konteks dokumen berikut.\n\n"
    f"Konteks dokumen:\n"
    f"---------------------\n"
    f"{context}\n"
    f"---------------------\n\n"
    f"Aturan WAJIB:\n"
    f"1. Jawab SELALU dalam Bahasa Indonesia\n"
    f"2. HANYA gunakan informasi dari konteks di atas\n"
    f"3. DILARANG menambahkan informasi dari luar konteks\n"
    f"4. DILARANG mengarang atau berasumsi\n"
    f"5. Jika pertanyaan meminta prosedur, sebutkan SEMUA langkah "
    f"yang ada di konteks secara berurutan\n"
    f"6. Jika informasi tidak ada di konteks, jawab dengan: "
    f"'Maaf, informasi tersebut tidak tersedia dalam dokumen internal kampus.'\n"
    f"7. Sertakan nomor form atau kode dokumen jika disebutkan di konteks\n\n"
    f"Pertanyaan: {question}\n"
    f"Jawaban berdasarkan konteks: "
    )

    full_answer = ""
    checked = False
    is_found = True

    print("[STREAMING] Mulai generate jawaban...")

    try:
        stream = ollama_client.generate(
            model=LLM_MODEL,
            prompt=prompt,
            stream=True
        )

        for chunk in stream:
            token = chunk.get("response", "")
            full_answer += token

            if not checked and len(full_answer) >= 50:
                checked = True
                answer_lower = full_answer.lower().strip()
                tidak_tahu = any(
                    phrase in answer_lower
                    for phrase in TIDAK_TAHU_TOKENS
                )
                if tidak_tahu:
                    print(f"[STREAMING] Deteksi dini: tidak tahu → stop")
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

    enhanced_question = question

    query_type = detect_query_type(enhanced_question)
    print(f"\n[DEBUG] Tipe query: {query_type}")

    # 3. Ambil konteks dari ChromaDB
    if index is None:
        response = query_engine.query(question)
        source_nodes = response.source_nodes
        context_text = str(response)
    else:
        source_nodes, context_text = retrieve_context(
            index, enhanced_question, query_type
        )

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

    # 4. Generate jawaban dengan interruptible streaming
    if index is not None:
        answer, is_found = stream_with_interrupt(enhanced_question, context_text)
    else:
        answer = str(response)
        is_found = True

    if not is_found:
        print("[DEBUG] Streaming dihentikan — jawaban tidak ada → HITL")
        return {"status": "not_found", "answer": None, "sources": []}

    # 5. Kumpulkan sumber
    sources_dict = {}
    for node in source_nodes:
        score = node.score if node.score else 0
        if score >= SIMILARITY_THRESHOLD:
            file_name = node.metadata.get("file_name", "Unknown")
            page = node.metadata.get("page_number", "?")
            tipe = node.metadata.get("tipe_dokumen", "")

            if tipe == "form":
                sources_dict[file_name] = f"{file_name} (formulir)"
            else:
                if file_name not in sources_dict:
                    sources_dict[file_name] = {
                        "name": file_name,
                        "pages": []
                    }
                if page not in sources_dict[file_name]["pages"]:
                    sources_dict[file_name]["pages"].append(page)

    # 6. Format sumber
    sources = []
    for key, val in sources_dict.items():
        if isinstance(val, str):
            sources.append(val)
        else:
            pages = ", ".join(str(p) for p in sorted(val["pages"]))
            sources.append(f"{val['name']} (hal. {pages})")

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