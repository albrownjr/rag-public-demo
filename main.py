import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import pipeline

from scripts.rag_engine import retrieve

load_dotenv()

app = FastAPI(title="RAG Public Demo", version="0.4.0")

# Load offline QA model once at startup
_qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Simple color lexicon for guardrail
_COLOR_WORDS = {
    "black","white","gray","grey","silver","red","blue","green","yellow","orange","brown",
    "beige","tan","gold","bronze","purple","pink","maroon","navy","teal"
}


class GenerateRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/")
def home():
    return {
        "status": "RAG Public Demo Running",
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/generate (POST)"],
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_color_terms(text: str) -> bool:
    t = _normalize(text)
    return any(w in t.split() for w in _COLOR_WORDS)


def _question_about_colors(query: str) -> bool:
    q = _normalize(query)
    # Basic intent detection
    return ("color" in q) or ("colours" in q) or ("paint" in q)


def _is_answer_grounded(answer: str, context: str) -> bool:
    a = _normalize(answer)
    c = _normalize(context)

    if not a:
        return False
    if len(a) < 4:
        return False

    return a in c


@app.post("/generate")
def generate(req: GenerateRequest):
    # 1️⃣ Retrieve relevant chunks
    hits = retrieve(req.query, k=req.top_k)

    # 2️⃣ Build combined context
    context_text = "\n".join([h["text"] for h in hits])

    # ✅ Guardrail: if the question is about colors but context has no color terms, refuse
    if _question_about_colors(req.query) and not _contains_color_terms(context_text):
        return {
            "mode": "offline_rag_refusal",
            "query": req.query,
            "answer": "I don't know based on the provided documents.",
            "reason": "color_question_but_no_color_evidence",
            "citations": hits,
        }

    # 3️⃣ Try OpenAI first (if key exists)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI()

            context_lines = [
                f"[{h['id']}] {h['text']} (source: {h['source']})"
                for h in hits
            ]
            context = "\n".join(context_lines)

            prompt = f"""
You are a RAG assistant.
Answer using ONLY the provided context.
If the answer is not present, say:
"I don't know based on the provided documents."

Include citation IDs like [1].

Context:
{context}

Question:
{req.query}
""".strip()

            resp = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5.2-codex"),
                input=prompt,
            )

            answer = (resp.output_text or "").strip()

            return {
                "mode": "rag_online",
                "query": req.query,
                "answer": answer,
                "citations": hits,
            }

        except Exception:
            pass  # Fall through to offline mode

    # 4️⃣ Offline QA fallback
    offline = _qa(question=req.query, context=context_text)

    answer = (offline.get("answer", "") or "").strip()
    score = float(offline.get("score", 0.0))

    CONFIDENCE_MIN = float(os.getenv("OFFLINE_CONFIDENCE_MIN", "0.30"))

    if score < CONFIDENCE_MIN:
        return {
            "mode": "offline_rag_refusal",
            "query": req.query,
            "answer": "I don't know based on the provided documents.",
            "confidence": score,
            "reason": "low_confidence",
            "citations": hits,
        }

    if not _is_answer_grounded(answer, context_text):
        return {
            "mode": "offline_rag_refusal",
            "query": req.query,
            "answer": "I don't know based on the provided documents.",
            "confidence": score,
            "reason": "not_grounded_in_context",
            "citations": hits,
        }

    return {
        "mode": "offline_rag",
        "query": req.query,
        "answer": answer,
        "confidence": score,
        "citations": hits,
    }