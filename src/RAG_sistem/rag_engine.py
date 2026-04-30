from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator
)
import chromadb
import ollama as ollama_client
import re

# ── Konfigurasi ───────────────────────────────────────────────────────────────
CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "stie_documents"
EMBED_MODEL = "qwen3-embedding"
LLM_MODEL = "qwen2.5:7b"
SIMILARITY_THRESHOLD = 0.4

# ── Token deteksi tidak tahu ──────────────────────────────────────────────────
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

# ── Kata kunci form ───────────────────────────────────────────────────────────
FORM_KEYWORDS = [
    "form apa",
    "formulir apa",
    "formulir untuk",
    "form untuk",
    "menggunakan form",
    "form yang digunakan",
    "formulir yang digunakan",
    "unduh form",
    "download form",
    "unduh formulir",
    "download formulir",
    "template",
    "blanko",
]

# ── Kata kunci relevansi kampus ───────────────────────────────────────────────
KAMPUS_KEYWORDS = [
    # Akademik
    "mahasiswa", "dosen", "kuliah", "akademik", "kampus", "stie", "prodi",
    "program studi", "semester", "sks", "nilai", "ujian", "sidang", "skripsi",
    "wisuda", "yudisium", "krs", "transkrip", "ipk", "mbkm", "mata kuliah",
    "kurikulum", "registrasi", "cuti", "drop out", "do", "nim", "nip",
    "tugas akhir", "ta", "tesis", "disertasi", "bimbingan",

    # Administrasi & Keuangan
    "dokumen", "form", "formulir", "pengajuan", "dana", "anggaran",
    "po", "spk", "rf", "requisition", "purchase order", "pembayaran",
    "reimburse", "kasbon", "invoice", "kwitansi", "tagihan", "keuangan",
    "finance", "anggaran", "transfer", "pencairan",

    # Departemen & Jabatan
    "baa", "bma", "lppm", "qa", "quality assurance", "hcm", "purchasing",
    "departemen", "informatika", "manajemen", "vcd", "visual",
    "head", "dekan", "kaprodi", "rektor", "direktur", "ketua",

    # Prosedur & Aturan
    "prosedur", "pedoman", "standar", "manual", "instruksi", "aturan",
    "kebijakan", "ketentuan", "syarat", "persyaratan", "mekanisme",
    "alur", "proses", "langkah", "tahap", "sop", "gui", "wi",

    # Fasilitas & Operasional
    "ruangan", "laboratorium", "perpustakaan", "parkir", "fasilitas",
    "absensi", "jadwal", "kalender akademik", "kegiatan", "event",

    # Dokumen spesifik
    "surat", "sk", "berita acara", "bast", "rr", "vo", "sipp",
    "pjk", "rpd", "bpu", "bg", "bilyet",
]

SAPAAN_KEYWORDS = [
    "halo", "hai", "hello", "hi", "hey", "hei",
    "selamat pagi", "selamat siang", "selamat sore",
    "selamat malam", "selamat datang", "permisi",
    "assalamu", "test", "tes", "coba",
]

PESAN_SAPAAN = (
    "Halo! Selamat datang di sistem tanya jawab dokumen internal "
    "STIE Ciputra Makassar. Silakan ajukan pertanyaan Anda "
    "seputar aturan, prosedur, atau pedoman kampus."
)

# ── Fungsi deteksi relevansi ──────────────────────────────────────────────────
def is_relevant_question(question: str) -> tuple[bool, str]:
    question_lower = question.lower().strip()

    PESAN_TIDAK_RELEVAN = (
        "Maaf, sistem ini hanya dapat menjawab pertanyaan seputar "
        "dokumen internal, aturan, prosedur, dan pedoman "
        "STIE Ciputra Makassar. Silakan ajukan pertanyaan yang "
        "berkaitan dengan kegiatan akademik atau administratif kampus."
    )

    # ── CEK 0: Deteksi sapaan ─────────────────────────────────────────────
    words = question_lower.split()
    first_word = words[0] if words else ""

    is_sapaan = (
        question_lower in SAPAAN_KEYWORDS
        or
        (first_word in SAPAAN_KEYWORDS and len(words) < 4)
    )

    if is_sapaan:
        return False, PESAN_SAPAAN

    # ── CEK 1: Ekspresi matematika murni ─────────────────────────────────
    cleaned = re.sub(
        r'\b(berapa|berapa hasil|hitung|hasil dari|berapakah)\b',
        '', question_lower
    ).strip()
    is_math = bool(re.match(
        r'^[\d\s\+\-\*\/\=\(\)\.\,\?]+$', cleaned
    ))
    if is_math:
        return False, PESAN_TIDAK_RELEVAN

    # ── CEK 2: Wajib mengandung keyword kampus ────────────────────────────
    has_keyword = any(
        keyword in question_lower
        for keyword in KAMPUS_KEYWORDS
    )
    if has_keyword:
        return True, None

    return False, PESAN_TIDAK_RELEVAN

# ── Fungsi deteksi tipe query ─────────────────────────────────────────────────
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


# ── Load index ────────────────────────────────────────────────────────────────
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


# ── Query engine ──────────────────────────────────────────────────────────────
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


# ── Retrieve context ──────────────────────────────────────────────────────────
def retrieve_context(
    index, question: str, query_type: str
) -> tuple[list, str]:
    """
    Ambil konteks dokumen berdasarkan tipe query.
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


# ── Streaming dengan interupsi ────────────────────────────────────────────────
def stream_with_interrupt(question: str, context: str) -> tuple[str, bool]:
    """
    Generate jawaban dengan interruptible streaming.
    Pantau 50 karakter pertama — kalau LLM tidak tahu, hentikan.
    """
    prompt = (
        f"Kamu adalah asisten sistem tanya jawab dokumen internal "
        f"STIE Ciputra Makassar.\n\n"
        f"KAMUS ISTILAH RESMI STIE CIPUTRA MAKASSAR "
        f"PERHATIAN: Gunakan kepanjangan PERSIS seperti di bawah ini.\n"
        f"- RF = Requisition Form (BUKAN Request Form, BUKAN Form Rekap Fee)\n"
        f"- Form Rekap Fee = dokumen terpisah dari RF, bukan singkatan\n"
        f"- PO = Purchase Order\n"
        f"- SPK = Surat Perjanjian Kerjasama\n"
        f"- RR = Receiving Record\n"
        f"- VO = Variation Order\n"
        f"- BAST = Berita Acara Serah Terima\n"
        f"- SIPP = Surat Instruksi Pelaksanaan Pekerjaan\n"
        f"- BPU = Bukti Pengeluaran Uang\n"
        f"- PJK = Pertanggungjawaban Keuangan\n"
        f"- RPD = Rincian Perjalanan Dinas\n"
        f"- BAA = Biro Administrasi Akademik\n"
        f"- BMA = Biro Mahasiswa dan Alumni\n"
        f"- LPPM = Lembaga Penelitian dan Pengabdian Masyarakat\n"
        f"- SKS = Satuan Kredit Semester\n"
        f"- MBKM = Merdeka Belajar Kampus Merdeka\n"
        f"- PDDIKTI = Pangkalan Data Pendidikan Tinggi\n"
        f"- QA = Quality Assurance\n\n"
        f"PENTING — CARA MEMBACA TABEL:\n"
        f"- Tabel 4 = 'Kelengkapan Dokumen Pendukung RF' → "
        f"untuk pertanyaan tentang dokumen pengajuan dana (kasbon/reimburse/tagihan)\n"
        f"- Tabel 5 = 'Dokumen Pendukung Laporan Pertanggungjawaban Keuangan (PJK)' → "
        f"untuk pertanyaan tentang laporan pertanggungjawaban setelah dana digunakan\n"
        f"- Jika pertanyaan tentang pengajuan dana → gunakan HANYA Tabel 4\n"
        f"- Jika pertanyaan tentang laporan PJK → gunakan HANYA Tabel 5\n"
        f"- Baca label kolom tabel dengan teliti sebelum menjawab\n\n"
        f"Konteks dokumen:\n"
        f"---------------------\n"
        f"{context}\n"
        f"---------------------\n\n"
        f"Aturan WAJIB — BACA DENGAN TELITI:\n"
        f"1. Jawab SELALU dalam Bahasa Indonesia\n"
        f"2. HANYA gunakan kalimat dan informasi yang ADA di konteks di atas\n"
        f"3. DILARANG KERAS menambah, mengarang, atau berasumsi "
        f"informasi yang tidak ada di konteks\n"
        f"4. Jika pertanyaan meminta prosedur, kutip langkah-langkah "
        f"PERSIS seperti yang tertulis di konteks, tidak lebih tidak kurang\n"
        f"5. Jawab HANYA untuk yang ditanyakan — "
        f"jangan tambahkan informasi untuk kategori lain "
        f"yang tidak disebutkan dalam pertanyaan\n"
        f"6. Gunakan kepanjangan singkatan PERSIS seperti di KAMUS ISTILAH di atas\n"
        f"7. Jika ada nomor form atau kode dokumen di konteks, sertakan\n"
        f"8. Jika informasi tidak ada di konteks, jawab dengan: "
        f"'Maaf, informasi tersebut tidak tersedia dalam dokumen internal kampus.'\n\n"
        f"9. DILARANG menyebutkan dari tabel mana atau bagian mana "
        f"informasi diambil dalam jawaban — cukup berikan jawabannya saja\n"
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
            stream=True,
            options={
                "temperature": 0,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
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
                    print("[STREAMING] Deteksi dini: tidak tahu → stop")
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


# ── Query documents ───────────────────────────────────────────────────────────
def query_documents(query_engine, question: str, index=None) -> dict:
    """
    Kirim pertanyaan ke sistem RAG dengan interruptible streaming.
    """

    # 1. Validasi relevansi pertanyaan
    is_relevant, pesan_tidak_relevan = is_relevant_question(question)
    if not is_relevant:
        return {
            "status": "not_relevant",
            "answer": pesan_tidak_relevan,
            "sources": []
        }

    # 2. Deteksi tipe query
    query_type = detect_query_type(question)
    print(f"\n[DEBUG] Tipe query: {query_type}")

    # 3. Ambil konteks dari ChromaDB
    if index is None:
        response = query_engine.query(question)
        source_nodes = response.source_nodes
        context_text = str(response)
    else:
        source_nodes, context_text = retrieve_context(
            index, question, query_type
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

    # 4. Khusus query form — langsung return info download tanpa generate
    if query_type == "form":
        form_files = []
        seen = set()
        for node in source_nodes:
            score = node.score if node.score else 0
            if score >= SIMILARITY_THRESHOLD:
                file_name = node.metadata.get("file_name", "Unknown")
                if file_name not in seen:
                    seen.add(file_name)
                    form_files.append({
                        "label": file_name,
                        "filename": file_name,
                        "is_form": True
                    })

        if form_files:
            return {
                "status": "found",
                "answer": (
                    "Berikut adalah formulir yang relevan dengan "
                    "pertanyaan kamu. Klik Preview untuk melihat "
                    "isi formulir atau Download untuk mengunduhnya."
                ),
                "sources": form_files,
                "is_form_response": True
            }

    # 5. Query naratif — generate jawaban dengan streaming
    if index is not None:
        answer, is_found = stream_with_interrupt(question, context_text)
    else:
        answer = str(response)
        is_found = True

    if not is_found:
        print("[DEBUG] Streaming dihentikan — jawaban tidak ada → HITL")
        return {"status": "not_found", "answer": None, "sources": []}

    # 6. Kumpulkan sumber dokumen
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

    # 7. Format sumber
    sources = []

    # Cek apakah ada sumber HITL
    if "hitl" in sources_dict:
        sources.append("Jawaban dari Staf QA")
    else:
        # Tampilkan sumber dokumen biasa
        for key, val in sources_dict.items():
            if isinstance(val, str):
                sources.append(val)
            else:
                pages = ", ".join(str(p) for p in sorted(val["pages"]))
                sources.append(f"{val['name']} (hal. {pages})")
    return {
        "status": "found",
        "answer": answer,
        "sources": sources,
        "is_form_response": False
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Memuat index dari ChromaDB...")
    index = load_index()
    query_engine = get_query_engine(index)
    print("Index berhasil dimuat!\n")

    while True:
        question = input(
            "Masukkan pertanyaan (ketik 'keluar' untuk berhenti): "
        )
        if question.lower() == "keluar":
            break

        print("\nMencari jawaban...")
        result = query_documents(query_engine, question, index=index)

        if result["status"] == "found":
            print(f"\nJawaban: {result['answer']}")
            if result.get("is_form_response"):
                print("Form tersedia:")
                for f in result["sources"]:
                    print(f"  - {f['label']}")
            else:
                print(f"Sumber: {', '.join(result['sources'])}")
        elif result["status"] == "not_relevant":
            print(f"\n{result['answer']}")
        else:
            print("\nInformasi tidak ditemukan dalam dokumen.")
            print("Pertanyaan ini akan diteruskan ke staf QA (HITL).")
        print("\n" + "=" * 50 + "\n")