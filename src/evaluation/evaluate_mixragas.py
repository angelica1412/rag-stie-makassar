# Evaluasi menggunakan RAGAS dan perhitungan secara manual 

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.run_config import RunConfig

from src.RAG_sistem.rag_engine import (
    load_index,
    get_query_engine,
    retrieve_context,
    stream_with_interrupt,
    detect_query_type
)
from src.evaluation.test_dataset import TEST_DATASET

EVAL_LLM_MODEL = "mistral:7b-instruct"
EVAL_EMBED_MODEL = "qwen3-embedding"

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "adalah", "untuk",
    "dengan", "pada", "ini", "itu", "atau", "juga", "dalam",
    "oleh", "tidak", "akan", "sudah", "ada", "bisa", "lebih",
    "saya", "kamu", "anda", "mereka", "kami", "kita", "bahwa",
    "jika", "maka", "telah", "dapat", "harus", "setelah", "sebelum"
}


def collect_rag_results(index, query_engine) -> list[dict]:
    results = []
    print(f"\nMemproses {len(TEST_DATASET)} pertanyaan...\n")

    for i, item in enumerate(TEST_DATASET):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i+1}/{len(TEST_DATASET)}] {question[:60]}...")

        try:
            query_type = detect_query_type(question)
            nodes, context_text = retrieve_context(
                index, question, query_type
            )
            contexts = [node.get_content() for node in nodes]
            answer, is_found = stream_with_interrupt(question, context_text)

            if not is_found or not answer.strip():
                answer = (
                    "Maaf, informasi tersebut tidak tersedia "
                    "dalam dokumen internal kampus."
                )

            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })
            print(f"  OK: {answer[:60]}...")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "question": question,
                "answer": "Error.",
                "contexts": [],
                "ground_truth": ground_truth
            })

    return results


def eval_context_precision_ragas(results: list[dict]) -> float:
    """Evaluasi Context Precision via RAGAS."""
    print("\n[RAGAS] Mengevaluasi Context Precision...")
    try:
        llm = LangchainLLMWrapper(
            ChatOllama(model=EVAL_LLM_MODEL, temperature=0, timeout=300)
        )
        context_precision.llm = llm

        dataset = Dataset.from_list(results)
        run_config = RunConfig(max_workers=1, max_retries=3, timeout=300)

        result = evaluate(
            dataset=dataset,
            metrics=[context_precision],
            raise_exceptions=False,
            run_config=run_config,
        )

        score = result['context_precision']
        if isinstance(score, list):
            valid = [s for s in score if s is not None]
            score = sum(valid) / len(valid) if valid else 0.0
        score = float(score) if score else 0.0

        import math
        if math.isnan(score):
            score = 0.0

        print(f"  Context Precision: {score:.4f}")
        return round(score, 4)

    except Exception as e:
        print(f"  Error: {e}")
        return 0.0


def eval_answer_relevancy_ragas(results: list[dict]) -> float:
    """Evaluasi Answer Relevancy via RAGAS."""
    print("\n[RAGAS] Mengevaluasi Answer Relevancy...")
    try:
        llm = LangchainLLMWrapper(
            ChatOllama(model=EVAL_LLM_MODEL, temperature=0, timeout=300)
        )
        embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=EVAL_EMBED_MODEL)
        )
        answer_relevancy.llm = llm
        answer_relevancy.embeddings = embeddings

        dataset = Dataset.from_list(results)
        run_config = RunConfig(max_workers=1, max_retries=3, timeout=300)

        result = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy],
            raise_exceptions=False,
            run_config=run_config,
        )

        score = result['answer_relevancy']
        if isinstance(score, list):
            valid = [s for s in score if s is not None]
            score = sum(valid) / len(valid) if valid else 0.0
        score = float(score) if score else 0.0

        import math
        if math.isnan(score):
            score = 0.0

        print(f"  Answer Relevancy: {score:.4f}")
        return round(score, 4)

    except Exception as e:
        print(f"  Error: {e}")
        return 0.0


def eval_context_recall_manual(results: list[dict]) -> float:
    """Context Recall — keyword matching."""
    print("\n[Manual] Menghitung Context Recall...")
    scores = []
    for r in results:
        gt_keywords = set(r["ground_truth"].lower().split()) - STOPWORDS
        if not gt_keywords:
            scores.append(0.0)
            continue
        combined = " ".join(r["contexts"]).lower()
        found = sum(1 for kw in gt_keywords if kw in combined)
        scores.append(found / len(gt_keywords))

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"  Context Recall: {avg:.4f}")
    return round(avg, 4)


def eval_faithfulness_manual(results: list[dict]) -> float:
    """Faithfulness — keyword matching."""
    print("\n[Manual] Menghitung Faithfulness...")
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

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"  Faithfulness: {avg:.4f}")
    return round(avg, 4)


if __name__ == "__main__":
    print("Memuat RAG index...")
    index = load_index()
    query_engine = get_query_engine(index)
    print("Index siap!\n")

    # 1. Kumpulkan hasil RAG
    results = collect_rag_results(index, query_engine)

    # 2. Evaluasi — RAGAS untuk CP dan AR, manual untuk CR dan F
    cp = eval_context_precision_ragas(results)
    ar = eval_answer_relevancy_ragas(results)
    cr = eval_context_recall_manual(results)
    f  = eval_faithfulness_manual(results)

    avg = round((cp + cr + f + ar) / 4, 4)

    # 3. Tampilkan hasil
    print("\n" + "="*60)
    print("HASIL EVALUASI SISTEM RAG")
    print("="*60)
    print(f"Context Precision  : {cp:.4f}  (RAGAS)")
    print(f"Context Recall     : {cr:.4f}  (Manual)")
    print(f"Faithfulness       : {f:.4f}  (Manual)")
    print(f"Answer Relevancy   : {ar:.4f}  (RAGAS)")
    print("="*60)
    print(f"Rata-rata          : {avg:.4f}")
    print("="*60)

    # 4. Simpan hasil
    final = {
        "model_sistem": "qwen2.5:7b",
        "model_evaluator": EVAL_LLM_MODEL,
        "total_pertanyaan": len(results),
        "scores": {
            "context_precision": cp,
            "context_recall": cr,
            "faithfulness": f,
            "answer_relevancy": ar,
            "average": avg
        },
        "detail": results
    }

    output_path = "./src/evaluation/hasil_evaluasi.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(final, out, ensure_ascii=False, indent=2)
    print(f"\nHasil disimpan ke: {output_path}")