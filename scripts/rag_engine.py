from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "public_demo"

# Load once (fast at runtime)
_model = SentenceTransformer("all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve(query: str, k: int = 3) -> list[dict]:
    """
    Returns a list of dicts:
    [
      {"id": "...", "text": "...", "source": "..."}
    ]
    """
    q_emb = _model.encode([query], normalize_embeddings=True).tolist()[0]
    res = _collection.query(query_embeddings=[q_emb], n_results=k)

    docs = res["documents"][0]
    ids = res["ids"][0]
    metas = res.get("metadatas", [[]])[0] or [{} for _ in docs]

    out = []
    for i in range(len(docs)):
        out.append(
            {
                "id": ids[i],
                "text": docs[i],
                "source": (metas[i] or {}).get("source", "unknown"),
            }
        )
    return out