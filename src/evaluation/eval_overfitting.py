import json
import sys
import os
sys.path.insert(0, '.')
from src.RAG_sistem.rag_engine import (
    load_index, retrieve_context,
    detect_query_type, query_documents,
    get_query_engine
)

# ── Dataset A — pertanyaan dari dataset evaluasi RAGAS ────────────────────────
dataset_evaluasi = [
    "Bagaimana proses transfer SKS MBKM?",
    "Apa yang dimaksud dengan Purchase Order?",
    "Dokumen apa yang diperlukan untuk kasbon dosen tamu?",
    "Apa itu Requisition Form?",
    "Berapa besaran retensi minimal SPK?",
]

# ── Dataset B — pertanyaan baru (makna sama, kata berbeda) ────────────────────
dataset_baru = [
    "Apa langkah pengakuan SKS mahasiswa magang merdeka?",
    "Jelaskan definisi dari Purchase Order!",
    "Apa saja berkas yang dibutuhkan untuk reimburse dosen tamu?",
    "Apa fungsi dari Requisition Form?",
    "Berapa persen retensi minimum dalam kontrak SPK?",
]

def hitung_rata_skor(pertanyaan_list, index):
    scores = []
    for q in pertanyaan_list:
        qt = detect_query_type(q)
        nodes, _ = retrieve_context(index, q, qt)
        if nodes and nodes[0].score:
            scores.append(nodes[0].score)
            print(f"  Skor: {nodes[0].score:.4f} | {q[:55]}")
        else:
            scores.append(0)
            print(f"  Skor: 0.0000 | {q[:55]}")
    return sum(scores) / len(scores) if scores else 0

# ── Jalankan evaluasi ─────────────────────────────────────────────────────────
print("Memuat index...")
index = load_index()

print("\n=== DATASET EVALUASI (pertanyaan lama) ===")
rata_lama = hitung_rata_skor(dataset_evaluasi, index)
print(f"Rata-rata skor: {rata_lama:.4f}")

print("\n=== DATASET BARU (pertanyaan baru) ===")
rata_baru = hitung_rata_skor(dataset_baru, index)
print(f"Rata-rata skor: {rata_baru:.4f}")

print("\n=== HASIL PERBANDINGAN ===")
selisih = abs(rata_lama - rata_baru)
print(f"Rata-rata dataset evaluasi : {rata_lama:.4f}")
print(f"Rata-rata dataset baru     : {rata_baru:.4f}")
print(f"Selisih                    : {selisih:.4f}")

if selisih < 0.05:
    print("Kesimpulan: Sistem TIDAK overfitting")
    print("           Performa konsisten untuk pertanyaan baru")
elif selisih < 0.10:
    print("Kesimpulan: Sedikit penurunan performa — masih wajar")
else:
    print("Kesimpulan: Indikasi overfitting — performa turun")
    print("           pada pertanyaan yang belum pernah diuji")

# Simpan hasil
hasil = {
    "rata_dataset_evaluasi": round(rata_lama, 4),
    "rata_dataset_baru": round(rata_baru, 4),
    "selisih": round(selisih, 4),
    "kesimpulan": (
        "Tidak overfitting" if selisih < 0.05
        else "Sedikit penurunan" if selisih < 0.10
        else "Indikasi overfitting"
    )
}

with open('./src/evaluation/hasil_overfitting.json', 'w') as f:
    json.dump(hasil, f, indent=2, ensure_ascii=False)

print("\nHasil disimpan: src/evaluation/hasil_overfitting.json")