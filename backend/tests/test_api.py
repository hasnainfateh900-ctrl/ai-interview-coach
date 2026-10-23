import os
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ.pop("ANTHROPIC_API_KEY", None)  # forces deterministic mock LLM
from fastapi.testclient import TestClient
from app.main import app
from app.pipeline import overall, readiness

c = TestClient(app)

def test_scoring_is_computed_in_code():
    assert overall({"correctness": 5, "depth": 5, "clarity": 5, "relevance": 5}) == 100
    assert overall({"correctness": 3, "depth": 3, "clarity": 3, "relevance": 3}) == 60
    assert readiness(80) == "Strong" and readiness(40) == "Needs work"

def test_full_interview_flow_and_isolation():
    import uuid
    h = {"Authorization": "Bearer " + c.post("/auth/register", json={"email": f"{uuid.uuid4()}@x.com", "password": "password123"}).json()["token"]}
    resume = "Built a Flask API with PostgreSQL and a React dashboard. " * 3
    rid = c.post("/resumes", files={"file": ("r.txt", resume.encode())}, headers=h).json()["id"]
    iv = c.post("/interviews", json={"resume_id": rid, "num_questions": 3}, headers=h).json()
    for _ in range(3):
        out = c.post(f"/interviews/{iv['id']}/answer", json={"answer": "word " * 100}, headers=h).json()
    assert out["status"] == "complete" and out["report"]["overall"] > 0
    assert c.get("/interviews/1").status_code in (401, 403)  # auth required

import uuid
from app.llm import parse_json
from app import rag

def signup():
    t = c.post("/auth/register", json={"email": f"{uuid.uuid4()}@x.com", "password": "password123"}).json()["token"]
    return {"Authorization": f"Bearer {t}"}

def new_interview(h, n=3):
    rid = c.post("/resumes", files={"file": ("r.txt", ("Built a Flask API with PostgreSQL. " * 4).encode())}, headers=h).json()["id"]
    return c.post("/interviews", json={"resume_id": rid, "num_questions": n}, headers=h).json()

def test_parse_json_tolerates_prose_and_braces():
    assert parse_json('Sure! ```json\n{"a": {"b": 1}}\n``` hope {this} helps') == {"a": {"b": 1}}

def test_user_cannot_read_another_users_interview():
    iv = new_interview(signup())
    assert c.get(f"/interviews/{iv['id']}", headers=signup()).status_code == 404

def test_weak_answer_triggers_followup_on_same_category():
    h = signup(); iv = new_interview(h)
    out = c.post(f"/interviews/{iv['id']}/answer", json={"answer": "idk"}, headers=h).json()
    assert out["current"]["category"] == iv["current"]["category"]  # probed, not rotated

def test_invalid_pdf_rejected():
    r = c.post("/resumes", files={"file": ("r.pdf", b"not a pdf")}, headers=signup())
    assert r.status_code == 422

def test_retrieval_prefers_relevant_chunk():
    q = rag.embed("PostgreSQL performance")
    sims = [sum(a * b for a, b in zip(rag.embed(t), q)) for t in ["Tuned PostgreSQL indexes", "Captain of soccer club"]]
    assert sims[0] > sims[1]
