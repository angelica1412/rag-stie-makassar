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
    "maaf, informasi tersebut tidak tersedia",
    "maaf, saya tidak dapat menemukan",
    "informasi tersebut tidak tersedia dalam dokumen",
    "tidak ditemukan dalam dokumen internal",
    "i cannot",
    "i don't know",
    "no information available",
]

# ── Kata kunci form ───────────────────────────────────────────────────────────
FORM_KEYWORDS = [
    "form",
    "formulir",
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

PESAN_PEMBUKA = (
    "Silakan sampaikan pertanyaan Anda seputar dokumen internal, "
    "aturan, prosedur, atau pedoman STIE Ciputra Makassar. "
    "Saya siap membantu!"
)

# ── Fungsi deteksi relevansi ──────────────────────────────────────────────────
# def is_relevant_question_llm(question: str) -> tuple[bool, str]:
#     prompt = (
#         f"Kamu adalah sistem validasi pertanyaan untuk layanan "
#         f"tanya jawab dokumen internal STIE Ciputra Makassar.\n\n"
#         f"Klasifikasikan input berikut ke dalam salah satu kategori:\n"
#         f"1. PERTANYAAN_VALID - pertanyaan lengkap dan relevan dengan "
#         f"konteks kampus\n"
#         f"2. SAPAAN - hanya berisi sapaan seperti halo, hai, selamat pagi\n"
#         f"3. PEMBUKA - kalimat pembuka tanpa pertanyaan seperti "
#         f"'saya ingin bertanya', 'mau tanya', 'permisi'\n"
#         f"4. TIDAK_RELEVAN - pertanyaan lengkap tapi tidak berkaitan "
#         f"dengan konteks kampus\n\n"
#         f"Jawab HANYA dengan satu kata: "
#         f"PERTANYAAN_VALID, SAPAAN, PEMBUKA, atau TIDAK_RELEVAN\n\n"
#         f"Input: {question}\n"
#         f"Kategori:"
#     )

#     try:
#         response = ollama_client.generate(
#             model=LLM_MODEL,
#             prompt=prompt,
#             stream=False,
#             options={"temperature": 0, "num_predict": 10}
#         )
#         result = response.get("response", "").strip().upper()
#         print(f"[DEBUG] Validasi LLM: {result}")

#         if "PERTANYAAN_VALID" in result:
#             return True, None
#         elif "SAPAAN" in result:
#             return False, PESAN_SAPAAN
#         elif "PEMBUKA" in result:
#             return False, PESAN_PEMBUKA
#         else:
#             return False, PESAN_TIDAK_RELEVAN

#     except Exception as e:
#         print(f"[DEBUG] Validasi LLM error: {e} → fallback")
#         return _fallback_validation(question)

# ── Fungsi deteksi tipe query ─────────────────────────────────────────────────
def detect_query_type(question: str) -> str:
    question_lower = question.lower()

    is_form_query = (
        any(keyword in question_lower for keyword in FORM_KEYWORDS)
        and not any(word in question_lower for word in [
            "informasi", "jelaskan", "apa itu", 
            "bagaimana", "prosedur", "syarat"
        ])
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

def _fallback_validation(question: str) -> tuple[bool, str]:
    """Fallback ke keyword matching kalau LLM tidak tersedia."""
    question_lower = question.lower().strip()

    PESAN_TIDAK_RELEVAN = (
        "Maaf, sistem ini hanya dapat menjawab pertanyaan seputar "
        "dokumen internal, aturan, prosedur, dan pedoman "
        "STIE Ciputra Makassar."
    )

    # Cek sapaan
    words = question_lower.split()
    first_word = words[0] if words else ""
    if (question_lower in SAPAAN_KEYWORDS or
            (first_word in SAPAAN_KEYWORDS and len(words) < 4)):
        return False, PESAN_SAPAAN

    # Cek matematika
    cleaned = re.sub(
        r'\b(berapa|hitung|hasil dari|berapakah)\b',
        '', question_lower
    ).strip()
    if bool(re.match(r'^[\d\s\+\-\*\/\=\(\)\.\,\?]+$', cleaned)):
        return False, PESAN_TIDAK_RELEVAN

    # Cek keyword kampus
    if any(kw in question_lower for kw in KAMPUS_KEYWORDS):
        return True, None

    return False, PESAN_TIDAK_RELEVAN

# ── Fungsi deteksi relevansi ──────────────────────────────────────────────────
def is_relevant_question_llm(question: str) -> tuple[bool, str]:
    """
    Gunakan LLM untuk menilai relevansi dan kelengkapan pertanyaan.
    """
    prompt = (
        f"Kamu adalah sistem validasi pertanyaan untuk layanan "
        f"tanya jawab dokumen internal STIE Ciputra Makassar.\n\n"
        f"Klasifikasikan input berikut ke dalam salah satu kategori:\n"
        f"1. PERTANYAAN_VALID - pertanyaan lengkap dan relevan dengan "
        f"konteks kampus (akademik, administrasi, prosedur, dokumen, "
        f"formulir, aturan kampus)\n"
        f"2. BUKAN_PERTANYAAN - kalimat pembuka, sapaan, atau "
        f"pernyataan tanpa pertanyaan yang jelas\n"
        f"3. TIDAK_RELEVAN - pertanyaan lengkap tapi tidak berkaitan "
        f"dengan konteks kampus\n\n"
        f"Jawab HANYA dengan satu kata: "
        f"PERTANYAAN_VALID, BUKAN_PERTANYAAN, atau TIDAK_RELEVAN\n\n"
        f"Input: {question}\n"
        f"Kategori:"
    )

    try:
        response = ollama_client.generate(
            model=LLM_MODEL,
            prompt=prompt,
            stream=False,
            options={"temperature": 0, "num_predict": 10}
        )
        result = response.get("response", "").strip().upper()
        print(f"[DEBUG] Validasi LLM: {result}")

        if "PERTANYAAN_VALID" in result:
            return True, None
        elif "BUKAN_PERTANYAAN" in result:
            return False, (
                "Silakan sampaikan pertanyaan Anda seputar dokumen "
                "internal, aturan, prosedur, atau pedoman STIE "
                "Ciputra Makassar. Saya siap membantu!"
            )
        else:
            return False, (
                "Maaf, sistem ini hanya dapat menjawab pertanyaan "
                "seputar dokumen internal STIE Ciputra Makassar."
            )

    except Exception as e:
        print(f"[DEBUG] Validasi LLM error: {e} → fallback")
        return _fallback_validation(question)

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
    "3. Jika informasi SAMA SEKALI tidak ada dalam konteks, jawab PERSIS dengan: "
    "'Maaf, informasi tersebut tidak tersedia dalam dokumen internal kampus.'\n"
    "4. Jangan mengarang jawaban\n"
    "5. Jangan menyarankan untuk mencari di tempat lain\n"
    "6. Jika ada nomor form, kode dokumen, atau istilah teknis "
    "yang disebutkan dalam konteks, sertakan dalam jawaban\n"
    "7. Jawab secara terstruktur dan lengkap\n"
    "8. Jika kamu sudah memberikan jawaban dari konteks, "
    "JANGAN tambahkan kalimat 'Maaf' atau 'tidak tersedia' "
    "di akhir jawaban. Akhiri jawaban langsung setelah informasi "
    "selesai disampaikan.\n\n"
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

def clean_answer(answer: str) -> str:
    """
    Hapus kalimat 'Maaf' yang ditambahkan LLM
    di akhir jawaban yang sebenarnya sudah lengkap.
    """
    # Frasa yang perlu dihapus kalau muncul di akhir
    trailing_phrases = [
        "Maaf, informasi tersebut tidak tersedia dalam dokumen internal kampus.",
        "Maaf, informasi tersebut tidak tersedia dalam dokumen.",
        "Maaf, saya tidak dapat menemukan informasi",
        "Informasi tersebut tidak tersedia dalam dokumen internal kampus.",
    ]
    
    cleaned = answer.strip()
    
    for phrase in trailing_phrases:
        if cleaned.endswith(phrase):
            cleaned = cleaned[:-len(phrase)].strip()
            break
    
    # Hapus juga kalau muncul setelah baris kosong di akhir
    lines = cleaned.split('\n')
    while lines and any(
        phrase.lower() in lines[-1].lower()
        for phrase in [
            "maaf, informasi tersebut",
            "tidak tersedia dalam dokumen internal",
            "informasi tersebut tidak tersedia"
        ]
    ):
        lines.pop()
    
    return '\n'.join(lines).strip()


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

    full_answer = clean_answer(full_answer)
    return full_answer, is_found


# ── Query documents ───────────────────────────────────────────────────────────
def query_documents(query_engine, question: str, index=None) -> dict:
    """
    Kirim pertanyaan ke sistem RAG dengan interruptible streaming.
    """

    is_relevant, pesan_tidak_relevan = is_relevant_question_llm(question)
    if not is_relevant:
        return {
            "status": "not_relevant",
            "answer": pesan_tidak_relevan,
            "sources": []
        }

    query_type = detect_query_type(question)
    print(f"\n[DEBUG] Tipe query: {query_type}")

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
        page  = node.metadata.get('page_number', '-')
        ctype = node.metadata.get('chunk_type', '-')
        print(f"[DEBUG] Node {i+1}: skor={score:.4f} | hal={page} | tipe={ctype} | file={fname}")
        print(f"        Isi: {node.get_content()[:500].strip()}...")
    print(f"[DEBUG] Top score: {top_score:.4f}, Threshold: {SIMILARITY_THRESHOLD}")

    if top_score < SIMILARITY_THRESHOLD:
        print("[DEBUG] Skor di bawah threshold → HITL")
        return {"status": "not_found", "answer": None, "sources": []}

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