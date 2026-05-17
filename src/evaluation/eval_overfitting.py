"""
Evaluasi Generalisasi Sistem RAG — Deteksi Overfitting

Tujuan: Membandingkan performa sistem RAG pada:
  - Dataset A : pertanyaan yang digunakan selama pengembangan/evaluasi utama
  - Dataset B : pertanyaan baru dengan topik sama tapi sudut pandang berbeda

Metrik yang diukur (lebih lengkap dari sekadar retrieval score):
  1. answer_similarity  — kemiripan cosine antara jawaban RAG dan ground truth
                          (pakai embedding qwen3-embedding)
  2. found_rate         — persentase pertanyaan yang berhasil dijawab sistem
                          (bukan HITL / not_found)
  3. top_retrieval_score — skor similarity tertinggi dari node yang diambil

Cara membaca hasilnya:
  - Selisih answer_similarity  < 0.05 → TIDAK overfitting
  - Selisih answer_similarity  < 0.10 → sedikit turun, masih wajar
  - Selisih answer_similarity >= 0.10 → indikasi overfitting
"""

import sys
import os
import json
import math
import numpy as np
sys.path.insert(0, '.')

from src.RAG_sistem.rag_engine import (
    load_index, retrieve_context,
    detect_query_type, query_documents,
    get_query_engine
)
from src.evaluation.overfitting_dataset import DATASET_A, DATASET_B

# ── Konfigurasi ───────────────────────────────────────────────────────────────
EMBED_MODEL = "qwen3-embedding"


# ── Helper embedding & cosine similarity ─────────────────────────────────────

def get_embedding(text: str, embed_model) -> list[float]:
    """Ambil embedding dari teks menggunakan model Ollama."""
    try:
        return embed_model.get_text_embedding(text)
    except Exception as e:
        print(f"  [WARN] Gagal embed: {e}")
        return []


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Hitung cosine similarity antara dua vektor embedding."""
    if not vec_a or not vec_b:
        return 0.0
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Fungsi evaluasi satu dataset ─────────────────────────────────────────────

def evaluate_dataset(dataset: list[dict], index, query_engine, embed_model, label: str) -> dict:
    """
    Evaluasi satu dataset.
    Return dict berisi skor per pertanyaan dan agregat.
    """
    print(f"\n{'='*60}")
    print(f"=== {label} ({len(dataset)} pertanyaan) ===")
    print(f"{'='*60}")

    per_kategori = {}
    all_answer_sim = []
    all_retrieval_scores = []
    found_count = 0

    for i, item in enumerate(dataset):
        question    = item["question"]
        ground_truth = item["ground_truth"]
        kategori    = item.get("kategori", "umum")

        print(f"\n[{i+1}/{len(dataset)}] [{kategori}] {question[:65]}")

        # ── 1. Jalankan pipeline RAG ──────────────────────────────────────
        try:
            result = query_documents(query_engine, question, index=index)
        except Exception as e:
            print(f"  Error: {e}")
            result = {"status": "error", "answer": ""}

        status = result.get("status", "error")
        answer = result.get("answer", "") or ""
        is_found = status == "found" and answer and "tidak tersedia" not in answer.lower()

        if is_found:
            found_count += 1

        # ── 2. Retrieval score ────────────────────────────────────────────
        try:
            qt = detect_query_type(question)
            nodes, _ = retrieve_context(index, question, qt)
            top_score = nodes[0].score if (nodes and nodes[0].score) else 0.0
        except Exception:
            top_score = 0.0
        all_retrieval_scores.append(top_score)

        # ── 3. Answer similarity vs ground truth ──────────────────────────
        if is_found and answer.strip():
            emb_answer = get_embedding(answer, embed_model)
            emb_gt     = get_embedding(ground_truth, embed_model)
            sim = cosine_similarity(emb_answer, emb_gt)
        else:
            # Jawaban tidak ditemukan → similarity 0 (penalti)
            sim = 0.0

        all_answer_sim.append(sim)

        # ── 4. Simpan per kategori ────────────────────────────────────────
        if kategori not in per_kategori:
            per_kategori[kategori] = []
        per_kategori[kategori].append(sim)

        print(f"  Status  : {status} | Found: {is_found}")
        print(f"  Retrieval score : {top_score:.4f}")
        print(f"  Answer sim (vs GT): {sim:.4f}")
        print(f"  Jawaban : {answer[:100]}...")

    # ── Agregat ───────────────────────────────────────────────────────────
    n = len(dataset)
    avg_answer_sim     = round(sum(all_answer_sim) / n, 4) if n else 0.0
    avg_retrieval      = round(sum(all_retrieval_scores) / n, 4) if n else 0.0
    found_rate         = round(found_count / n, 4) if n else 0.0

    print(f"\n--- Ringkasan {label} ---")
    print(f"  Answer Similarity (rata-rata) : {avg_answer_sim:.4f}")
    print(f"  Retrieval Score (rata-rata)   : {avg_retrieval:.4f}")
    print(f"  Found Rate                    : {found_rate:.4f} ({found_count}/{n})")

    # Skor per kategori
    print(f"\n  Skor per kategori (answer similarity):")
    for kat, sims in per_kategori.items():
        avg_kat = round(sum(sims) / len(sims), 4)
        print(f"    {kat:<25} : {avg_kat:.4f}  ({len(sims)} pertanyaan)")

    return {
        "label"             : label,
        "total"             : n,
        "found_count"       : found_count,
        "found_rate"        : found_rate,
        "avg_answer_sim"    : avg_answer_sim,
        "avg_retrieval"     : avg_retrieval,
        "per_kategori"      : {k: round(sum(v)/len(v), 4) for k, v in per_kategori.items()},
        "detail"            : [
            {
                "question"       : dataset[j]["question"],
                "kategori"       : dataset[j].get("kategori", "umum"),
                "answer_sim"     : all_answer_sim[j],
                "retrieval_score": all_retrieval_scores[j],
            }
            for j in range(n)
        ]
    }


# ── Kesimpulan ────────────────────────────────────────────────────────────────

def buat_kesimpulan(hasil_a: dict, hasil_b: dict) -> str:
    selisih_sim      = hasil_a["avg_answer_sim"] - hasil_b["avg_answer_sim"]
    selisih_found    = hasil_a["found_rate"]     - hasil_b["found_rate"]

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append("HASIL EVALUASI GENERALISASI (OVERFITTING CHECK)")
    lines.append(f"{'='*60}")
    lines.append(f"  {'Metrik':<35} {'Dataset A':>10}  {'Dataset B':>10}  {'Selisih':>8}")
    lines.append(f"  {'-'*65}")
    lines.append(
        f"  {'Answer Similarity (vs ground truth)':<35} "
        f"{hasil_a['avg_answer_sim']:>10.4f}  "
        f"{hasil_b['avg_answer_sim']:>10.4f}  "
        f"{selisih_sim:>+8.4f}"
    )
    lines.append(
        f"  {'Retrieval Score (embedding sim)':<35} "
        f"{hasil_a['avg_retrieval']:>10.4f}  "
        f"{hasil_b['avg_retrieval']:>10.4f}  "
        f"{hasil_a['avg_retrieval']-hasil_b['avg_retrieval']:>+8.4f}"
    )
    lines.append(
        f"  {'Found Rate':<35} "
        f"{hasil_a['found_rate']:>10.4f}  "
        f"{hasil_b['found_rate']:>10.4f}  "
        f"{selisih_found:>+8.4f}"
    )
    lines.append(f"{'='*60}")

    # Kesimpulan berdasarkan answer similarity
    if selisih_sim < 0.05:
        status = "TIDAK OVERFITTING"
        detail = (
            "Performa sistem KONSISTEN untuk pertanyaan baru.\n"
            "Sistem berhasil menggeneralisasi pengetahuannya."
        )
    elif selisih_sim < 0.10:
        status = "SEDIKIT PENURUNAN — MASIH WAJAR"
        detail = (
            "Performa sedikit turun pada pertanyaan baru.\n"
            "Ini masih dalam batas normal untuk sistem RAG."
        )
    else:
        status = "INDIKASI OVERFITTING"
        detail = (
            "Performa turun signifikan pada pertanyaan baru.\n"
            "Sistem kemungkinan terlalu dikonfigurasi untuk dataset evaluasi.\n"
            "Pertimbangkan: perluas test dataset, tuning threshold, atau\n"
            "perbaiki chunking agar lebih general."
        )

    lines.append(f"\nKesimpulan : {status}")
    lines.append(f"Keterangan :\n  {detail}")

    # Analisis per kategori
    lines.append(f"\nAnalisis per kategori:")
    all_kats = set(hasil_a["per_kategori"].keys()) | set(hasil_b["per_kategori"].keys())
    for kat in sorted(all_kats):
        sim_a = hasil_a["per_kategori"].get(kat, None)
        sim_b = hasil_b["per_kategori"].get(kat, None)
        if sim_a is not None and sim_b is not None:
            diff = sim_a - sim_b
            flag = " ⚠️" if diff > 0.10 else ""
            lines.append(f"  {kat:<25} A={sim_a:.4f}  B={sim_b:.4f}  Δ={diff:+.4f}{flag}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from llama_index.embeddings.ollama import OllamaEmbedding

    print("Memuat index dan embedding model...")
    index        = load_index()
    query_engine = get_query_engine(index)
    embed_model  = OllamaEmbedding(model_name=EMBED_MODEL)
    print("Siap!\n")

    # Evaluasi Dataset A (referensi)
    hasil_a = evaluate_dataset(DATASET_A, index, query_engine, embed_model, "Dataset A (Referensi)")

    # Evaluasi Dataset B (generalisasi)
    hasil_b = evaluate_dataset(DATASET_B, index, query_engine, embed_model, "Dataset B (Generalisasi)")

    # Kesimpulan
    kesimpulan = buat_kesimpulan(hasil_a, hasil_b)
    print(kesimpulan)

    # Simpan hasil
    output = {
        "keterangan": (
            "Dataset A = pertanyaan referensi dari evaluasi utama. "
            "Dataset B = pertanyaan baru dengan topik sama tapi sudut pandang berbeda. "
            "Metrik utama: answer_similarity (cosine similarity jawaban RAG vs ground truth)."
        ),
        "dataset_a": hasil_a,
        "dataset_b": hasil_b,
        "perbandingan": {
            "selisih_answer_sim"   : round(hasil_a["avg_answer_sim"] - hasil_b["avg_answer_sim"], 4),
            "selisih_retrieval"    : round(hasil_a["avg_retrieval"]   - hasil_b["avg_retrieval"], 4),
            "selisih_found_rate"   : round(hasil_a["found_rate"]      - hasil_b["found_rate"], 4),
            "kesimpulan"           : (
                "Tidak overfitting"    if (hasil_a["avg_answer_sim"] - hasil_b["avg_answer_sim"]) < 0.05
                else "Sedikit penurunan" if (hasil_a["avg_answer_sim"] - hasil_b["avg_answer_sim"]) < 0.10
                else "Indikasi overfitting"
            )
        }
    }

    output_path = "./src/evaluation/hasil_overfitting.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nHasil disimpan: {output_path}")