"""Resume RAG: chunk -> embed -> store in pgvector -> retrieve top-k for each question.
Default embedder is a signed hashing bag-of-words (lexical, offline, zero deps).
Set EMBED_MODEL=all-MiniLM-L6-v2 (and pip install sentence-transformers) for semantic embeddings; both are 384-dim."""
import hashlib, math, os, re
from sqlalchemy import select
from .models import Chunk, DIM

_model = None

def embed(text: str) -> list[float]:
    global _model
    if os.getenv("EMBED_MODEL"):
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(os.environ["EMBED_MODEL"])
        return _model.encode(text, normalize_embeddings=True).tolist()
    v = [0.0] * DIM
    for w in re.findall(r"[a-z0-9+#.]+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM] += 1 if (h >> 20) & 1 else -1
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

def chunk(text: str, size: int = 500) -> list[str]:
    out, cur = [], ""
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        if cur and len(cur) + len(line) > size:
            out.append(cur); cur = ""
        cur = (cur + "\n" + line).strip()
    return out + [cur] if cur else out

def retrieve(db, resume_id: int, query: str, k: int = 3) -> list[str]:
    q = embed(query)
    base = select(Chunk).where(Chunk.resume_id == resume_id)
    if db.bind.dialect.name == "postgresql":
        rows = db.scalars(base.order_by(Chunk.embedding.cosine_distance(q)).limit(k)).all()
    else:  # sqlite dev/test fallback
        rows = sorted(db.scalars(base).all(), key=lambda c: -sum(a * b for a, b in zip(c.embedding, q)))[:k]
    return [c.text for c in rows]
