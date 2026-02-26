from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sample_knowledge.csv"

CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "public_demo"


def main():
    print("Loading CSV:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain columns: id,text")

    texts = df["text"].astype(str).tolist()
    ids = df["id"].astype(str).tolist()

    print(f"Loaded {len(texts)} rows")

    print("Loading embedding model (first time may take a minute)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding documents...")
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    print("Creating / opening Chroma DB at:", CHROMA_PATH)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection:", COLLECTION_NAME)
    except Exception:
        pass

    col = client.create_collection(name=COLLECTION_NAME)

    print("Adding docs to Chroma...")
    col.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": "sample_knowledge.csv"} for _ in texts],
    )

    print("✅ Ingest complete.")
    print(f"Collection '{COLLECTION_NAME}' now has {col.count()} documents.")

    query = "Do you offer 0% financing?"
    print("\nTest query:", query)
    q_emb = model.encode([query], normalize_embeddings=True).tolist()[0]
    results = col.query(query_embeddings=[q_emb], n_results=2)

    print("\nTop matches:")
    for i, doc in enumerate(results["documents"][0], start=1):
        print(f"{i}. {doc}")


if __name__ == "__main__":
    main()
