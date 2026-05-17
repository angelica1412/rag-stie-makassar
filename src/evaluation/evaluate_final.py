import sys
import os
import json
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.RAG_sistem.rag_engine import (
    load_index,
    get_query_engine,
    query_documents,
    retrieve_context,
    detect_query_type
)
from src.evaluation.test_dataset import TEST_DATASET

# ── Konfigurasi ───────────────────────────────────────────────────────────────
EVAL_LLM_MODEL   = "llama3.1:8b"
EVAL_EMBED_MODEL = "qwen3-embedding"

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "adalah", "untuk",
    "dengan", "pada", "ini", "itu", "atau", "juga", "dalam",
    "oleh", "tidak", "akan", "sudah", "ada", "bisa", "lebih",
    "saya", "kamu", "anda", "mereka", "kami", "kita", "bahwa",
    "jika", "maka", "telah", "dapat", "harus", "setelah", "sebelum"
}


# ── Helper ────────────────────────────────────────────────────────────────────

def safe_score(value, default=0.0) -> float:
    """Ambil nilai score yang aman — ganti NaN dengan default."""
    if value is None:
        return default
    try:
        if isinstance(value, list):
            valid = [
                float(s) for s in value
                if s is not None and not math.isnan(float(s))
            ]
            return sum(valid) / len(valid) if valid else default
        val = float(value)
        return default if math.isnan(val) else val
    except Exception:
        return default


# ── Setup RAGAS ───────────────────────────────────────────────────────────────

def get_ragas_config():
    """Setup LLM dan embedding Ollama untuk RAGAS."""
    llm = LangchainLLMWrapper(
        ChatOllama(
            model=EVAL_LLM_MODEL,
            temperature=0,
            timeout=300,
            num_predict=1024,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EVAL_EMBED_MODEL)
    )
    return llm, embeddings


# ── Kumpulkan hasil RAG ───────────────────────────────────────────────────────

def collect_rag_results(index, query_engine) -> list[dict]:
    """
    Jalankan setiap pertanyaan ke sistem RAG menggunakan
    pipeline yang SAMA PERSIS dengan sistem produksi
    melalui query_documents().
    """
    results = []
    print(f"\nMemproses {len(TEST_DATASET)} pertanyaan...\n")

    for i, item in enumerate(TEST_DATASET):
        question     = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i+1}/{len(TEST_DATASET)}] {question[:60]}...")

        try:
            # ── Gunakan pipeline lengkap — sama dengan sistem produksi ──
            result = query_documents(query_engine, question, index=index)

            if result["status"] == "found":
                answer = result["answer"]
            else:
                # Status not_found atau not_relevant
                answer = (
                    "Maaf, informasi tersebut tidak tersedia "
                    "dalam dokumen internal kampus."
                )

            # ── Ambil contexts untuk RAGAS ───────────────────────────────
            # Contexts diambil ulang agar RAGAS bisa mengevaluasi
            # context_precision dan context_recall
            query_type = detect_query_type(question)
            nodes, _   = retrieve_context(index, question, query_type)
            contexts   = [node.get_content() for node in nodes]

            results.append({
                "question"    : question,
                "answer"      : answer,
                "contexts"    : contexts,
                "ground_truth": ground_truth
            })

            print(f"  Status : {result['status']}")
            print(f"  Jawaban: {answer[:80]}...")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "question"    : question,
                "answer"      : "Error saat memproses pertanyaan.",
                "contexts"    : [],
                "ground_truth": ground_truth
            })

    return results


# ── Evaluasi RAGAS ────────────────────────────────────────────────────────────

def run_ragas_evaluation(results: list[dict]) -> dict:
    """
    Jalankan semua metrik RAGAS dengan llama3.1:8b.
    Menggunakan cara yang kompatibel dengan Ollama local:
    import singleton dari ragas.metrics, assign metric.llm = llm,
    lalu evaluate per metrik satu per satu.
    """
    print(f"\nMenjalankan evaluasi RAGAS dengan {EVAL_LLM_MODEL}...")

    llm, embeddings = get_ragas_config()
    run_config = RunConfig(
        max_workers=1,
        max_retries=5,
        timeout=300,
    )

    dataset = Dataset.from_list(results)
    ragas_scores = {}

    metrics_to_run = [
        ("context_precision",  context_precision),
        ("context_recall",     context_recall),
        ("faithfulness",       faithfulness),
        ("answer_relevancy",   answer_relevancy),
    ]

    for metric_name, metric in metrics_to_run:
        print(f"\nMengevaluasi {metric_name}...")
        try:
            # Assign LLM ke metric (cara yang kompatibel dengan Ollama)
            metric.llm = llm
            if metric_name == "answer_relevancy":
                metric.embeddings = embeddings

            result = evaluate(
                dataset=dataset,
                metrics=[metric],
                raise_exceptions=False,
                run_config=run_config,
            )

            # Gunakan bracket notation (result[key]) bukan result.get(key)
            try:
                score = safe_score(result[metric_name])
            except (KeyError, TypeError):
                score = None

            ragas_scores[metric_name] = round(score, 4) if score is not None else None
            print(f"  {metric_name}: {ragas_scores[metric_name]}")

        except Exception as e:
            print(f"  Error pada {metric_name}: {e}")
            ragas_scores[metric_name] = None

    return ragas_scores


# ── Evaluasi manual fallback ──────────────────────────────────────────────────

def eval_context_recall_manual(results: list[dict]) -> float:
    """
    Context Recall manual — fallback jika RAGAS gagal.
    Mengukur seberapa banyak informasi dari ground_truth
    yang tercakup di dalam retrieved contexts.
    Rumus: keywords ground_truth yang ditemukan di contexts / total keywords ground_truth
    """
    scores = []
    for r in results:
        gt_keywords = set(r["ground_truth"].lower().split()) - STOPWORDS
        if not gt_keywords:
            scores.append(0.0)
            continue
        combined = " ".join(r["contexts"]).lower()
        found = sum(1 for kw in gt_keywords if kw in combined)
        scores.append(found / len(gt_keywords))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def eval_context_precision_manual(results: list[dict]) -> float:
    """
    Context Precision manual — fallback jika RAGAS gagal.
    Mengukur berapa fraksi dari retrieved contexts yang benar-benar relevan
    terhadap ground truth (bukan noise).
    Rumus: untuk tiap context, hitung apakah mengandung ≥30% keyword ground_truth.
           Precision = jumlah context relevan / total context yang diambil.
    """
    scores = []
    for r in results:
        gt_keywords = set(r["ground_truth"].lower().split()) - STOPWORDS
        if not gt_keywords or not r["contexts"]:
            scores.append(0.0)
            continue
        relevant_count = 0
        for ctx in r["contexts"]:
            ctx_lower = ctx.lower()
            found = sum(1 for kw in gt_keywords if kw in ctx_lower)
            # Context dianggap relevan jika mencakup >= 30% keyword ground_truth
            if found / len(gt_keywords) >= 0.30:
                relevant_count += 1
        scores.append(relevant_count / len(r["contexts"]))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def eval_faithfulness_manual(results: list[dict]) -> float:
    """
    Faithfulness manual — fallback jika RAGAS gagal.
    Mengukur seberapa banyak isi jawaban yang bisa ditemukan di contexts
    (jawaban tidak mengarang di luar konteks).
    Rumus: keywords jawaban yang ada di contexts / total keywords jawaban
    """
    scores = []
    for r in results:
        answer = r["answer"]
        if "tidak tersedia" in answer.lower():
            scores.append(1.0)
            continue
        ans_keywords = set(answer.lower().split()) - STOPWORDS
        if not ans_keywords:
            scores.append(0.0)
            continue
        combined = " ".join(r["contexts"]).lower()
        found = sum(1 for kw in ans_keywords if kw in combined)
        scores.append(found / len(ans_keywords))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def eval_answer_relevancy_manual(results: list[dict]) -> float:
    """
    Answer Relevancy manual — fallback jika RAGAS gagal.
    Mengukur seberapa relevan jawaban terhadap pertanyaan.
    Rumus: keyword pertanyaan yang muncul di jawaban / total keyword pertanyaan
    Catatan: ini pendekatan keyword-based; RAGAS asli menggunakan embedding.
    """
    scores = []
    for r in results:
        answer = r["answer"]
        if "tidak tersedia" in answer.lower() or not answer.strip():
            # Jawaban tidak tersedia = tidak relevan
            scores.append(0.0)
            continue
        q_keywords = set(r["question"].lower().split()) - STOPWORDS
        if not q_keywords:
            scores.append(0.0)
            continue
        ans_lower = answer.lower()
        found = sum(1 for kw in q_keywords if kw in ans_lower)
        scores.append(found / len(q_keywords))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Memuat RAG index...")
    index        = load_index()
    query_engine = get_query_engine(index)
    print("Index siap!\n")

    # 1. Kumpulkan hasil RAG menggunakan pipeline produksi
    results = collect_rag_results(index, query_engine)

    # 2. Jalankan evaluasi RAGAS
    ragas_scores = run_ragas_evaluation(results)

    # 3. Tentukan skor final (RAGAS atau fallback manual)
    final_scores = {}

    for metric in ["context_precision", "context_recall",
                   "faithfulness", "answer_relevancy"]:
        ragas_val = ragas_scores.get(metric)  # bisa None atau float

        if ragas_val is not None and not math.isnan(float(ragas_val)):
            # RAGAS berhasil — gunakan skor dari RAGAS
            final_scores[metric]             = round(float(ragas_val), 4)
            final_scores[f"{metric}_method"] = "RAGAS"
        else:
            # Fallback ke implementasi manual per metrik
            if metric == "context_precision":
                val = eval_context_precision_manual(results)
            elif metric == "context_recall":
                val = eval_context_recall_manual(results)
            elif metric == "faithfulness":
                val = eval_faithfulness_manual(results)
            elif metric == "answer_relevancy":
                val = eval_answer_relevancy_manual(results)
            else:
                val = 0.0
            final_scores[metric]             = val
            final_scores[f"{metric}_method"] = "Manual"

    # 4. Hitung rata-rata
    avg = round((
        final_scores["context_precision"] +
        final_scores["context_recall"]    +
        final_scores["faithfulness"]      +
        final_scores["answer_relevancy"]
    ) / 4, 4)
    final_scores["average"] = avg

    # 5. Tampilkan hasil
    print("\n" + "=" * 60)
    print("HASIL EVALUASI SISTEM RAG")
    print("=" * 60)
    print(
        f"Context Precision  : "
        f"{final_scores['context_precision']:.4f}"
        f"  [{final_scores['context_precision_method']}]"
    )
    print(
        f"Context Recall     : "
        f"{final_scores['context_recall']:.4f}"
        f"  [{final_scores['context_recall_method']}]"
    )
    print(
        f"Faithfulness       : "
        f"{final_scores['faithfulness']:.4f}"
        f"  [{final_scores['faithfulness_method']}]"
    )
    print(
        f"Answer Relevancy   : "
        f"{final_scores['answer_relevancy']:.4f}"
        f"  [{final_scores['answer_relevancy_method']}]"
    )
    print("=" * 60)
    print(f"Rata-rata          : {avg:.4f}")
    print("=" * 60)

    # 6. Tampilkan detail per pertanyaan
    print("\n── Detail Jawaban Per Pertanyaan ──")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r['question'][:70]}")
        print(f"     Jawaban: {r['answer'][:100]}...")

    # 7. Simpan hasil
    output = {
        "model_sistem"    : "qwen2.5:7b",
        "model_evaluator" : EVAL_LLM_MODEL,
        "total_pertanyaan": len(results),
        "scores": {
            "context_precision": final_scores["context_precision"],
            "context_recall"   : final_scores["context_recall"],
            "faithfulness"     : final_scores["faithfulness"],
            "answer_relevancy" : final_scores["answer_relevancy"],
            "average"          : avg
        },
        "metode": {
            "context_precision": final_scores["context_precision_method"],
            "context_recall"   : final_scores["context_recall_method"],
            "faithfulness"     : final_scores["faithfulness_method"],
            "answer_relevancy" : final_scores["answer_relevancy_method"],
        },
        "detail": results
    }

    output_path = "./src/evaluation/hasil_evaluasi (1).json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nHasil disimpan ke: {output_path}")