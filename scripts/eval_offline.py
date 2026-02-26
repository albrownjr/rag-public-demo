import json
from pathlib import Path

from scripts.rag_engine import retrieve
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "eval_results.json"

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

COLOR_WORDS = {
    "black","white","gray","grey","silver","red","blue","green","yellow","orange","brown",
    "beige","tan","gold","bronze","purple","pink","maroon","navy","teal"
}


def normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def contains_color_terms(text: str) -> bool:
    words = set(normalize(text).split())
    return any(c in words for c in COLOR_WORDS)


def question_about_colors(q: str) -> bool:
    qn = normalize(q)
    return ("color" in qn) or ("colours" in qn) or ("paint" in qn)


def run_case(query: str, expect_refusal: bool, top_k: int = 2):
    hits = retrieve(query, k=top_k)
    context_text = "\n".join([h["text"] for h in hits])

    # Guardrail: color question but no evidence
    if question_about_colors(query) and not contains_color_terms(context_text):
        return {
            "query": query,
            "mode": "offline_rag_refusal",
            "reason": "color_question_but_no_color_evidence",
            "answer": "I don't know based on the provided documents.",
            "citations": hits,
            "pass": expect_refusal is True,
        }

    offline = qa(question=query, context=context_text)
    answer = (offline.get("answer", "") or "").strip()
    score = float(offline.get("score", 0.0))

    CONFIDENCE_MIN = 0.30

    if score < CONFIDENCE_MIN or not answer:
        return {
            "query": query,
            "mode": "offline_rag_refusal",
            "reason": "low_confidence",
            "answer": "I don't know based on the provided documents.",
            "confidence": score,
            "citations": hits,
            "pass": expect_refusal is True,
        }

    if normalize(answer) not in normalize(context_text):
        return {
            "query": query,
            "mode": "offline_rag_refusal",
            "reason": "not_grounded_in_context",
            "answer": "I don't know based on the provided documents.",
            "confidence": score,
            "citations": hits,
            "pass": expect_refusal is True,
        }

    return {
        "query": query,
        "mode": "offline_rag",
        "answer": answer,
        "confidence": score,
        "citations": hits,
        "pass": expect_refusal is False,
    }


def print_table(results: list[dict]):
    # Simple readable table
    print("\nRESULTS")
    print("-" * 90)
    print(f"{'PASS':<6} {'MODE':<22} {'CONF':<6}  QUERY")
    print("-" * 90)
    for r in results:
        p = "YES" if r["pass"] else "NO"
        mode = r.get("mode", "")
        conf = r.get("confidence", "")
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else ""
        q = r["query"]
        print(f"{p:<6} {mode:<22} {conf_str:<6}  {q}")
        if not r["pass"]:
            print(f"       ↳ reason: {r.get('reason','')}")
            print(f"       ↳ answer: {r.get('answer','')}")
    print("-" * 90)


def main():
    test_cases = [
        # Should answer from docs
        {"query": "Do you offer 0% financing?", "expect_refusal": False},
        {"query": "What trims are available for the 2025 Sonata?", "expect_refusal": False},
        {"query": "Should I schedule a test drive in advance?", "expect_refusal": False},

        # Should refuse (not in docs)
        {"query": "What colors does the 2025 Sonata come in?", "expect_refusal": True},
        {"query": "Does the Sonata have a panoramic roof?", "expect_refusal": True},
        {"query": "What is the towing capacity of the 2025 Sonata?", "expect_refusal": True},
        {"query": "Is there a hybrid Sonata option mentioned here?", "expect_refusal": True},
    ]

    results = [run_case(**c) for c in test_cases]
    passed = sum(1 for r in results if r["pass"])
    total = len(results)

    summary = {
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 3),
        "results": results,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"✅ Eval complete: {passed}/{total} passed (pass_rate={summary['pass_rate']})")
    print(f"Saved: {OUT_PATH}")

    print_table(results)


if __name__ == "__main__":
    main()