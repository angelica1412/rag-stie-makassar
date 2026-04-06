import os
import re
import fitz 
import pdfplumber
from llama_index.core import Document


def clean_text(text: str) -> str:
    """
    Bersihkan teks hasil ekstraksi PDF:
    - Normalkan multiple spasi dalam satu baris
    - Hapus baris kosong berlebihan
    """
    # Normalkan spasi berlebihan dalam satu baris
    text = re.sub(r" {2,}", " ", text)
    # Hapus baris kosong berlebihan (lebih dari 2 berturut-turut)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_tables_from_page(plumber_page) -> str:
    """
    Ekstrak tabel dari halaman menggunakan pdfplumber (lebih baik untuk tabel).
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


def read_pdfs_from_folder(folder_path: str) -> list[Document]:
    documents = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue

        filepath = os.path.join(folder_path, filename)
        print(f"Membaca: {filename}")

        try:
            # Buka dengan PyMuPDF untuk ekstraksi teks
            fitz_doc = fitz.open(filepath)
            # Buka dengan pdfplumber untuk ekstraksi tabel
            plumber_doc = pdfplumber.open(filepath)

            page_count = 0

            for page_num in range(len(fitz_doc)):
                # Ekstrak teks dengan PyMuPDF (spasi terjaga dengan benar)
                fitz_page = fitz_doc[page_num]
                text = fitz_page.get_text("text")
                text = clean_text(text)

                # Ekstrak tabel dengan pdfplumber
                plumber_page = plumber_doc.pages[page_num]
                table_text = extract_tables_from_page(plumber_page)

                # Gabungkan teks dan tabel
                page_content = text
                if table_text:
                    page_content += "\n[TABEL]\n" + table_text

                # Hanya buat dokumen jika ada konten
                if page_content.strip():
                    doc = Document(
                        text=page_content,
                        metadata={
                            "file_name": filename,
                            "file_path": filepath,
                            "source": filename,
                            "page_number": page_num + 1
                        }
                    )
                    documents.append(doc)
                    page_count += 1

            fitz_doc.close()
            plumber_doc.close()

            print(f"  Berhasil: {page_count} halaman diindeks dari {filename}")

        except Exception as e:
            print(f"  Error membaca {filename}: {e}")

    print(f"\nTotal dokumen (halaman) berhasil dibaca: {len(documents)}")
    return documents