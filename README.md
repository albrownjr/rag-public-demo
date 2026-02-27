# RAG Public Demo (Offline + Online Switch)

A production-style Retrieval-Augmented Generation (RAG) API that:
- **Ingests** a public CSV knowledge base into a **vector database (Chroma)**
- **Retrieves** relevant passages with **local embeddings** (sentence-transformers)
- **Answers** questions using:
  - **Offline mode** (no paid API required) via an extractive QA model
  - **Online mode** (optional) via OpenAI, if `OPENAI_API_KEY` is set
- Includes **guardrails** to reduce hallucinations:
  - Confidence threshold
  - Intent-aware refusal for “color” questions without evidence
- Includes an **evaluation suite** that runs test cases and outputs pass rate

## Why this stands out
Most RAG demos only “work.” This repo shows how to build RAG like a real system:
- retrieval + citations
- safety/refusal logic (“I don’t know”)
- repeatable evaluation with saved results
- CI automation (GitHub Actions)

---

## Quickstart (Local)

### 1) Create venv + install
```bash
python -m venv venv
# Windows (cmd)
venv\Scripts\activate.bat
pip install -r requirements.txt