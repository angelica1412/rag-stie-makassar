# Pembacaan Dokumen

import os
import re
import fitz
import pdfplumber
from llama_index.core import Document

# ── Path subfolder ────────────────────────────────────────────────────────────
NARATIF_PATH = "./data/documents/naratif"
FORM_PATH = "./data/documents/form"


def decode_filename(filename: str) -> str:
    """
    Decode nama file yang ter-URL-encode.
    Contoh: 'UC_20STD_20BAA_2003' → 'UC STD BAA 2003'
    '_20' adalah representasi dari spasi (%20) yang di-encode saat download.
    """
    return filename.replace("_20", " ")


def clean_text(text: str) -> str:
    """
    Bersihkan teks hasil ekstraksi PDF:
    - Normalkan multiple spasi dalam satu baris
    - Hapus baris kosong berlebihan
    """
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_tables_from_page(plumber_page) -> str:
    """
    Ekstrak tabel dari halaman menggunakan pdfplumber.
    """
    table_text = ""
    try:
        tables = plumber_page.extract_tables()
        for table in tables:
            for row in table:
                cleaned = [cell.strip() if cell else "" for cell in row]
                table_text += " | ".join(cleaned) + "\n"
    except Exception:
        pass
    return table_text


def read_naratif_documents() -> list[Document]:
    """
    Baca dokumen naratif (pedoman, standar, manual, IK) secara full-text
    termasuk tabel — sama persis dengan cara baca kamu sebelumnya.
    """
    documents = []
    print("\n=== Membaca dokumen NARATIF ===")

    if not os.path.exists(NARATIF_PATH):
        print(f"Folder tidak ditemukan: {NARATIF_PATH}")
        return documents

    for filename in os.listdir(NARATIF_PATH):
        if not filename.endswith(".pdf"):
            continue

        filepath = os.path.join(NARATIF_PATH, filename)
        print(f"Membaca: {filename}")

        try:
            fitz_doc = fitz.open(filepath)
            plumber_doc = pdfplumber.open(filepath)
            page_count = 0

            for page_num in range(len(fitz_doc)):
                fitz_page = fitz_doc[page_num]
                text = fitz_page.get_text("text")
                text = clean_text(text)

                plumber_page = plumber_doc.pages[page_num]
                table_text = extract_tables_from_page(plumber_page)

                page_content = text
                if table_text:
                    page_content += "\n[TABEL]\n" + table_text

                if page_content.strip():
                    doc = Document(
                        text=page_content,
                        metadata={
                            "file_name": decode_filename(filename),
                            "file_path": filepath,
                            "source": decode_filename(filename),
                            "page_number": page_num + 1,
                            "tipe_dokumen": "naratif"
                        }
                    )
                    documents.append(doc)
                    page_count += 1

            fitz_doc.close()
            plumber_doc.close()
            print(f"  Berhasil: {page_count} halaman dari {filename}")

        except Exception as e:
            print(f"  Error membaca {filename}: {e}")

    print(f"Total halaman naratif: {len(documents)} halaman")
    return documents


def read_form_documents() -> list[Document]:
    """
    Baca dokumen form — hanya metadata dari halaman pertama.
    Tidak membaca isi template atau tabel kosong sesuai batasan penelitian.
    """
    documents = []
    print("\n=== Membaca dokumen FORM (metadata only) ===")

    if not os.path.exists(FORM_PATH):
        print(f"Folder tidak ditemukan: {FORM_PATH}")
        return documents

    for filename in os.listdir(FORM_PATH):
        if not filename.endswith(".pdf"):
            continue

        filepath = os.path.join(FORM_PATH, filename)
        decoded_name = decode_filename(filename)
        print(f"Membaca form: {decoded_name}")

        try:
            fitz_doc = fitz.open(filepath)

            # Hanya baca halaman pertama untuk ambil metadata
            first_page = fitz_doc[0]
            raw_text = first_page.get_text("text")
            text = clean_text(raw_text)

            # Ambil maksimal 500 karakter pertama sebagai metadata
            metadata_text = text[:500] if len(text) > 500 else text

            # Buat deskripsi dokumen form
            doc_content = (
                f"Formulir: {decoded_name}\n"
                f"Tipe Dokumen: Formulir\n"
                f"Deskripsi: {metadata_text}\n"
                f"Kegunaan: Formulir ini tersedia untuk digunakan "
                f"sesuai kebutuhan akademik dan administratif kampus."
            )

            fitz_doc.close()

            if doc_content.strip():
                doc = Document(
                    text=doc_content,
                    metadata={
                        "file_name": decoded_name,
                        "file_path": filepath,
                        "source": decoded_name,
                        "page_number": 1,
                        "tipe_dokumen": "form"
                    }
                )
                documents.append(doc)
                print(f"  Berhasil: metadata form {decoded_name}")

        except Exception as e:
            print(f"  Error membaca form {filename}: {e}")

    print(f"Total form: {len(documents)} form")
    return documents


def read_pdfs_from_folder(folder_path: str = None) -> list[Document]:
    """
    Fungsi utama — baca semua dokumen dari kedua subfolder.
    - Naratif: full text + tabel (per halaman)
    - Form: metadata only (per file)
    """
    all_documents = []

    # Baca dokumen naratif
    naratif_docs = read_naratif_documents()
    all_documents.extend(naratif_docs)

    # Baca dokumen form
    form_docs = read_form_documents()
    all_documents.extend(form_docs)

    # Hitung jumlah file naratif unik
    naratif_files = set(
        doc.metadata.get("file_name", "") 
        for doc in naratif_docs
    )

    print(f"\n=== Ringkasan Ingestion ===")
    print(f"Dokumen naratif : {len(naratif_files)} file ({len(naratif_docs)} halaman)")
    print(f"Dokumen form    : {len(form_docs)} file")
    print(f"Total chunks    : {len(all_documents)}")
    
    return all_documents