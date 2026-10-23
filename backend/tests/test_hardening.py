import time, uuid, jwt, pytest
from types import SimpleNamespace as NS
from pydantic import ValidationError
from fastapi.testclient import TestClient
from app.main import app
from app import llm, pipeline, auth

c = TestClient(app)
GOOD = {"correctness": 3, "depth": 3, "clarity": 3, "relevance": 3, "feedback": "f", "ideal_points": []}

def signup():
    t = c.post("/auth/register", json={"email": f"{uuid.uuid4()}@x.com", "password": "password123"}).json()["token"]
    return {"Authorization": f"Bearer {t}"}

def interview(h, n=3):
    rid = c.post("/resumes", files={"file": ("r.txt", ("Built a Flask API with PostgreSQL. " * 4).encode())}, headers=h).json()["id"]
    return c.post("/interviews", json={"resume_id": rid, "num_questions": n}, headers=h).json()

def fake_client(replies):
    it = iter(replies)
    create = lambda **kw: NS(content=[NS(text=next(it))], usage=NS(input_tokens=10, output_tokens=5))
    return NS(messages=NS(create=create))

# --- auth
def test_duplicate_registration_and_weak_password():
    body = {"email": f"{uuid.uuid4()}@x.com", "password": "password123"}
    assert c.post("/auth/register", json=body).status_code == 200
    assert c.post("/auth/register", json=body).status_code == 409
    assert c.post("/auth/register", json={"email": "a@b.com", "password": "short"}).status_code == 422

def test_missing_invalid_and_expired_jwt():
    assert c.get("/interviews").status_code in (401, 403)
    assert c.get("/interviews", headers={"Authorization": "Bearer junk"}).status_code == 401
    expired = jwt.encode({"sub": "1", "exp": int(time.time()) - 10}, auth.SECRET, "HS256")
    assert c.get("/interviews", headers={"Authorization": f"Bearer {expired}"}).status_code == 401

# --- resumes
def test_oversized_and_empty_resume():
    h = signup()
    assert c.post("/resumes", files={"file": ("r.txt", b"x" * 2_000_001)}, headers=h).status_code == 413
    assert c.post("/resumes", files={"file": ("r.txt", b"  ")}, headers=h).status_code == 422
    assert c.post("/resumes", files={"file": ("r.exe", b"x" * 100)}, headers=h).status_code == 415

# --- interviews
def test_unknown_interview_and_duplicate_submission():
    h = signup()
    assert c.get("/interviews/999999", headers=h).status_code == 404
    iv = interview(h)
    for _ in range(3):
        c.post(f"/interviews/{iv['id']}/answer", json={"answer": "word " * 100}, headers=h)
    assert c.post(f"/interviews/{iv['id']}/answer", json={"answer": "again"}, headers=h).status_code == 409

# --- LLM failure modes
def test_retry_on_malformed_json_and_metrics_logged(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(llm, "_client", lambda: fake_client(["not json", '{"a": 1}']))
    assert llm.complete_json("s", "u", lambda: {}) == {"a": 1}
    m = c.get("/metrics", headers=signup()).json()
    assert m["calls"] >= 1 and m["retry_rate"] > 0

def test_gives_up_after_two_bad_replies(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(llm, "_client", lambda: fake_client(["nope", "still nope"]))
    with pytest.raises(RuntimeError):
        llm.complete_json("s", "u", lambda: {})

@pytest.mark.parametrize("bad", [{**GOOD, "correctness": 9}, {"correctness": 3}, {**GOOD, "depth": "high"}])
def test_invalid_scores_or_missing_fields_rejected(monkeypatch, bad):
    monkeypatch.setattr(pipeline, "complete_json", lambda *a, **k: bad)
    with pytest.raises(ValidationError):
        pipeline.evaluate("q", "technical", "a", {})

def test_api_returns_502_on_invalid_llm_output(monkeypatch):
    h = signup(); iv = interview(h)
    monkeypatch.setattr(pipeline, "complete_json", lambda *a, **k: {"correctness": 99})
    assert c.post(f"/interviews/{iv['id']}/answer", json={"answer": "hello"}, headers=h).status_code == 502
