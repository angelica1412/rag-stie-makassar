# Pengelolaan Dokumen

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from src.RAG_sistem.pdf_reader import read_pdfs_from_folder
import chromadb


DOCS_PATH = "./data/documents"
CHROMA_PATH = "./data/chroma_db"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "stie_documents"

# ── Konfigurasi Chunking ──────────────────────────────────────────────────────
# chunk_size   : jumlah karakter per chunk — lebih kecil = lebih fokus = skor lebih tinggi
# chunk_overlap: tumpang tindih antar chunk agar konteks tidak terpotong
CHUNK_SIZE    = 512   # ~3-5 kalimat
CHUNK_OVERLAP = 100   # ~1 kalimat tumpang tindih

def build_index():
    print("Memulai proses ingestion dokumen...")
    print(f"Konfigurasi chunking: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}\n")

    # 1. Set embedding model
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None

    # 2. Konfigurasi SentenceSplitter
    #    Memecah teks per kalimat dengan ukuran maksimal CHUNK_SIZE karakter
    #    Ini menghasilkan chunk yang lebih fokus dan relevan daripada per-halaman
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",  # pisahkan antar paragraf dulu
    )
    Settings.node_parser = splitter

    # 3. Baca PDF
    documents = read_pdfs_from_folder(DOCS_PATH)

    if not documents:
        print("Tidak ada dokumen yang berhasil dibaca!")
        return None

    print(f"\nTotal dokumen (halaman) dibaca: {len(documents)}")

    # 4. Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Buat index — SentenceSplitter otomatis dipakai oleh Settings.node_parser
    print("\nMemproses chunking + embedding... ini butuh beberapa menit")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
        transformations=[splitter],   # eksplisit pakai splitter ini
    )

    # Hitung estimasi jumlah chunk yang dibuat
    print(f"\n✅ Ingestion selesai!")
    print(f"   Chunk size   : {CHUNK_SIZE} karakter")
    print(f"   Chunk overlap: {CHUNK_OVERLAP} karakter")
    print(f"   Data tersimpan di: {CHROMA_PATH}")
    return index

if __name__ == "__main__":
    build_index()