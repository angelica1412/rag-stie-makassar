# Pembacaan Dokumen

import os
import re
import fitz
import pdfplumber
from llama_index.core import Document

from docx import Document as DocxDocument

import openpyxl

NARATIF_PATH = "./data/documents/naratif"
FORM_PATH = "./data/documents/form"


def decode_filename(filename: str) -> str:
    """Decode karakter URL-encoded dalam nama file."""
    # _20 = spasi (%20), _2F = slash (%2F), dll
    decoded = filename
    decoded = decoded.replace("_2F", "/")
    decoded = decoded.replace("_2B", "+")
    decoded = decoded.replace("_20", " ")
    return decoded


def clean_text(text: str) -> str:
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

HEADER_STOP_PATTERNS = [
    r"reproducing or copying without permission",
    r"uncontrolled copy",
    r"controlled document",
    r"document no[:\.]",
    r"revision[:\.]",
    r"effective date[:\.]",
    r"expired date[:\.]",
]

def remove_page_header(text: str) -> str:

    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        is_header_line = any(
            re.search(pattern, line_lower)
            for pattern in HEADER_STOP_PATTERNS
        )
        
        is_doc_code = bool(re.match(
            r'^[a-z]{2,5}[/\-][a-z]{2,5}[/\-]', 
            line_lower
        ))
        
        if not is_header_line and not is_doc_code:
            cleaned_lines.append(line)
    
    return clean_text('\n'.join(cleaned_lines))

def extract_tables_from_page(plumber_page, preceding_text: str = "") -> str:

    table_text = ""
    try:
        tables = plumber_page.extract_tables()
        for table_idx, table in enumerate(tables):
            if not table:
                continue

            # Cari nama tabel dari teks sebelumnya
            table_name = f"Tabel {table_idx + 1}"
            if preceding_text:
                lines = preceding_text.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line.lower().startswith('tabel') and len(line) < 150:
                        table_name = line
                        break

            # Ambil header dari baris pertama
            headers = [
                cell.strip() if cell else f"Kolom{i+1}"
                for i, cell in enumerate(table[0])
            ]

            table_text += f"\n[{table_name}]\n"
            table_text += f"Kolom: {' | '.join(headers)}\n"

            # Format setiap baris dengan label kolom
            for row_idx, row in enumerate(table[1:], start=1):
                if not any(cell for cell in row if cell):
                    continue

                table_text += f"\nBaris {row_idx}:\n"
                for col_idx, cell in enumerate(row):
                    if col_idx < len(headers):
                        header = headers[col_idx]
                        value = cell.strip() if cell else "-"
                        if value and value != "-":
                            table_text += f"  {header}: {value}\n"

    except Exception:
        pass
    return table_text

# ── Ekstraksi metadata per tipe file ─────────────────────────────────────────

def extract_metadata_from_pdf(filepath: str) -> str:
    """Ambil teks dari halaman pertama PDF."""
    try:
        fitz_doc = fitz.open(filepath)
        text = fitz_doc[0].get_text("text")
        fitz_doc.close()
        text = clean_text(text)
        return text[:500] if len(text) > 500 else text
    except Exception as e:
        return f"Tidak dapat membaca PDF: {e}"


def extract_metadata_from_docx(filepath: str) -> str:
    """Ambil teks dari paragraf pertama file docx."""
    try:
        doc = DocxDocument(filepath)
        texts = []
        for para in doc.paragraphs[:20]:  # ambil 20 paragraf pertama
            if para.text.strip():
                texts.append(para.text.strip())
        result = "\n".join(texts)
        return result[:500] if len(result) > 500 else result
    except Exception as e:
        return f"Tidak dapat membaca DOCX: {e}"


def extract_metadata_from_xlsx(filepath: str) -> str:
    """Ambil teks dari baris pertama file xlsx."""
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        ws = wb.active
        texts = []
        row_count = 0
        for row in ws.iter_rows(values_only=True):
            if row_count >= 10:  # ambil 10 baris pertama
                break
            row_text = " | ".join(
                str(cell) for cell in row if cell is not None
            )
            if row_text.strip():
                texts.append(row_text)
                row_count += 1
        wb.close()
        result = "\n".join(texts)
        return result[:500] if len(result) > 500 else result
    except Exception as e:
        return f"Tidak dapat membaca XLSX: {e}"


# ── Fungsi utama baca dokumen ─────────────────────────────────────────────────

def read_naratif_documents() -> list[Document]:
    documents = []
    print("\n=== Membaca dokumen NARATIF ===")

    if not os.path.exists(NARATIF_PATH):
        print(f"Folder tidak ditemukan: {NARATIF_PATH}")
        return documents

    for filename in os.listdir(NARATIF_PATH):
        if not filename.endswith(".pdf"):
            continue

        filepath = os.path.join(NARATIF_PATH, filename)
        display_name = decode_filename(filename)
        print(f"Membaca: {display_name}")

        try:
            fitz_doc = fitz.open(filepath)
            plumber_doc = pdfplumber.open(filepath)
            page_count = 0

            for page_num in range(len(fitz_doc)):
                fitz_page = fitz_doc[page_num]
                text = fitz_page.get_text("text")
                text = remove_page_header(text)
                
                plumber_page = plumber_doc.pages[page_num]

                # Cek apakah halaman mengandung tabel
                tables = []
                try:
                    tables = plumber_page.extract_tables()
                except Exception:
                    pass

                if tables:
                    # ── Halaman dengan tabel: buat chunk terpisah per tabel
                    
                    # Chunk 1: teks naratif halaman (tanpa tabel)
                    if text.strip():
                        doc = Document(
                            text=text,
                            metadata={
                                "file_name": display_name,
                                "file_path": filepath,
                                "source": display_name,
                                "page_number": page_num + 1,
                                "tipe_dokumen": "naratif",
                                "chunk_type": "text"
                            }
                        )
                        documents.append(doc)
                        page_count += 1

                    # Chunk 2+: satu chunk per tabel
                    for table_idx, table in enumerate(tables):
                        if not table:
                            continue

                        # Cari nama tabel dari teks sebelumnya
                        table_name = f"Tabel {table_idx + 1}"
                        lines = text.split('\n')
                        for line in reversed(lines):
                            line_stripped = line.strip()
                            if (line_stripped.lower().startswith('tabel')
                                    and len(line_stripped) < 150):
                                table_name = line_stripped
                                break

                        # Format tabel dengan label kolom
                        headers = [
                            cell.strip() if cell else f"Kolom{i+1}"
                            for i, cell in enumerate(table[0])
                        ]

                        table_content = f"[{table_name}]\n"
                        table_content += f"Kolom: {' | '.join(headers)}\n"

                        for row in table[1:]:
                            if not any(cell for cell in row if cell):
                                continue
                            table_content += "\n"
                            for col_idx, cell in enumerate(row):
                                if col_idx < len(headers):
                                    header = headers[col_idx]
                                    value = cell.strip() if cell else "-"
                                    if value and value != "-":
                                        table_content += (
                                            f"  {header}: {value}\n"
                                        )

                        if table_content.strip():
                            doc = Document(
                                text=table_content,
                                metadata={
                                    "file_name": display_name,
                                    "file_path": filepath,
                                    "source": display_name,
                                    "page_number": page_num + 1,
                                    "tipe_dokumen": "naratif",
                                    "chunk_type": "table",
                                    "table_name": table_name
                                }
                            )
                            documents.append(doc)
                            page_count += 1
                            print(
                                f"    Tabel ditemukan: {table_name}"
                            )

                else:
                    # ── Halaman tanpa tabel: chunk biasa
                    if text.strip():
                        doc = Document(
                            text=text,
                            metadata={
                                "file_name": display_name,
                                "file_path": filepath,
                                "source": display_name,
                                "page_number": page_num + 1,
                                "tipe_dokumen": "naratif",
                                "chunk_type": "text"
                            }
                        )
                        documents.append(doc)
                        page_count += 1

            fitz_doc.close()
            plumber_doc.close()
            print(f"  Berhasil: {page_count} chunks dari {display_name}")

        except Exception as e:
            print(f"  Error membaca {filename}: {e}")

    naratif_files = set(
        doc.metadata.get("file_name", "") for doc in documents
    )
    print(
        f"Total naratif: {len(naratif_files)} file "
        f"({len(documents)} chunks)"
    )
    return documents

def read_form_documents() -> list[Document]:
    """
    Baca dokumen form — metadata only.
    Mendukung: PDF, DOCX, DOC, XLSX.
    """
    documents = []
    print("\n=== Membaca dokumen FORM (metadata only) ===")

    if not os.path.exists(FORM_PATH):
        print(f"Folder tidak ditemukan: {FORM_PATH}")
        return documents

    # Format yang didukung
    supported_ext = (".pdf", ".docx", ".doc", ".xlsx")

    for filename in os.listdir(FORM_PATH):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_ext:
            continue

        filepath = os.path.join(FORM_PATH, filename)
        decoded_name = decode_filename(filename)
        print(f"Membaca form: {decoded_name}")

        try:
            # Ekstrak metadata sesuai tipe file
            if ext == ".pdf":
                metadata_text = extract_metadata_from_pdf(filepath)
                tipe_file = "PDF"
            elif ext in (".docx", ".doc"):
                metadata_text = extract_metadata_from_docx(filepath)
                tipe_file = "Word Document"
            elif ext == ".xlsx":
                metadata_text = extract_metadata_from_xlsx(filepath)
                tipe_file = "Excel Spreadsheet"
            else:
                continue

            # Buat deskripsi dokumen form
            doc_content = (
                f"Formulir: {decoded_name}\n"
                f"Tipe File: {tipe_file}\n"
                f"Tipe Dokumen: Formulir\n"
                f"Deskripsi: {metadata_text}\n"
                f"Kegunaan: Formulir ini tersedia untuk digunakan "
                f"sesuai kebutuhan akademik dan administratif kampus."
            )

            if doc_content.strip():
                doc = Document(
                    text=doc_content,
                    metadata={
                        "file_name": decoded_name,
                        "file_path": filepath,
                        "source": decoded_name,
                        "page_number": 1,
                        "tipe_dokumen": "form",
                        "tipe_file": tipe_file
                    }
                )
                documents.append(doc)
                print(f"  Berhasil: {tipe_file} — {decoded_name}")

        except Exception as e:
            print(f"  Error membaca {filename}: {e}")

    print(f"Total form: {len(documents)} file")
    return documents


def read_pdfs_from_folder(folder_path: str = None) -> list[Document]:
    """
    Fungsi utama — baca semua dokumen dari kedua subfolder.
    - Naratif: full text + tabel (PDF only)
    - Form: metadata only (PDF, DOCX, DOC, XLSX)
    """
    all_documents = []

    naratif_docs = read_naratif_documents()
    all_documents.extend(naratif_docs)

    form_docs = read_form_documents()
    all_documents.extend(form_docs)

    print(f"\n=== Ringkasan Ingestion ===")
    naratif_files = set(
        doc.metadata.get("file_name", "")
        for doc in naratif_docs
    )
    print(f"Dokumen naratif : {len(naratif_files)} file "
          f"({len(naratif_docs)} halaman)")
    print(f"Dokumen form    : {len(form_docs)} file")
    print(f"Total chunks    : {len(all_documents)}")

    return all_documents