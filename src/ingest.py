import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from src.pdf_reader import read_pdfs_from_folder
import chromadb


DOCS_PATH = "./data/documents"
CHROMA_PATH = "./data/chroma_db"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "stie_documents"

def build_index():
    print("Memulai proses ingestion dokumen...")

    # 1. Set embedding model
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None

    # 2. Baca PDF menggunakan pdfplumber
    documents = read_pdfs_from_folder(DOCS_PATH)

    if not documents:
        print("Tidak ada dokumen yang berhasil dibaca!")
        return None

    # 3. Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Buat index
    print("\nMemproses embedding... ini butuh beberapa menit")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print("\nIngestion selesai! Dokumen tersimpan di ChromaDB.")
    return index

if __name__ == "__main__":
    build_index()