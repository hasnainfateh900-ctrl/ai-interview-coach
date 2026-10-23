import io
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from fastapi.responses import JSONResponse
from statistics import mean
from sqlalchemy import select, func, text as sql
from sqlalchemy.orm import Session
from . import pipeline, rag, llm
from .auth import hash_pw, check_pw, make_token, current_user, limit
from .db import Base, engine, get_db
from .models import User, Resume, Interview, Turn, Chunk, LLMCall

if engine.dialect.name == "postgresql":
    with engine.begin() as c:
        c.execute(sql("CREATE EXTENSION IF NOT EXISTS vector"))
Base.metadata.create_all(engine)  # swap for Alembic migrations in production (schema changes need a fresh DB until then)
app = FastAPI(title="AI Interview Coach")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

class Creds(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)

class StartReq(BaseModel):
    resume_id: int
    num_questions: int = Field(5, ge=3, le=10)

class AnswerReq(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)

@app.post("/auth/register")
def register(c: Creds, db: Session = Depends(get_db)):
    email = c.email.lower().strip()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email already registered")
    u = User(email=email, password_hash=hash_pw(c.password))
    db.add(u); db.commit()
    return {"token": make_token(u.id)}

@app.post("/auth/login")
def login(c: Creds, db: Session = Depends(get_db)):
    u = db.scalar(select(User).where(User.email == c.email.lower().strip()))
    if not u or not check_pw(c.password, u.password_hash):
        raise HTTPException(401, "Wrong email or password")
    return {"token": make_token(u.id)}

@app.post("/resumes")
async def upload_resume(file: UploadFile = File(...), user: User = Depends(limit), db: Session = Depends(get_db)):
    data = await file.read()
    name = (file.filename or "").lower()
    if len(data) > 2_000_000:
        raise HTTPException(413, "File too large (2 MB max)")
    if not name.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(415, "Upload a .pdf, .txt or .md file")
    try:
        if name.endswith(".pdf"):
            if not data.startswith(b"%PDF-"):
                raise ValueError
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
        else:
            text = data.decode("utf-8")
    except Exception:
        raise HTTPException(422, "File is not a readable PDF or UTF-8 text document")
    text = text.strip()[:12000]
    if len(text) < 50:
        raise HTTPException(422, "Could not read enough text from this resume")
    r = Resume(user_id=user.id, text=text, analysis=pipeline.analyze_resume(text))
    db.add(r); db.flush()
    db.add_all(Chunk(resume_id=r.id, text=t, embedding=rag.embed(t)) for t in rag.chunk(text))
    db.commit()
    return {"id": r.id, "analysis": r.analysis}

def owned(db: Session, iid: int, user: User) -> Interview:
    iv = db.get(Interview, iid)
    if not iv or iv.user_id != user.id:
        raise HTTPException(404, "Interview not found")
    return iv

def view(iv: Interview) -> dict:
    done = [t for t in iv.turns if t.answer is not None]
    nxt = next((t for t in iv.turns if t.answer is None), None)
    return {
        "id": iv.id, "status": iv.status, "total": iv.total, "answered": len(done), "report": iv.report,
        "current": {"idx": nxt.idx, "category": nxt.category, "question": nxt.question} if nxt else None,
        "history": [{"idx": t.idx, "category": t.category, "question": t.question,
                     "answer": t.answer, "evaluation": t.evaluation} for t in done],
    }

@app.post("/interviews")
def start(req: StartReq, user: User = Depends(limit), db: Session = Depends(get_db)):
    r = db.get(Resume, req.resume_id)
    if not r or r.user_id != user.id:
        raise HTTPException(404, "Resume not found")
    cat = pipeline.CATEGORIES[0]
    excerpts = rag.retrieve(db, r.id, cat + " " + " ".join(r.analysis["weaknesses"]))
    q = pipeline.next_question(r.analysis, [], excerpts, cat)
    iv = Interview(user_id=user.id, resume_id=r.id, total=req.num_questions, turns=[Turn(idx=0, category=cat, question=q)])
    db.add(iv); db.commit()
    return view(iv)

@app.get("/interviews")
def history(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Interview).where(Interview.user_id == user.id).order_by(Interview.id.desc())).all()
    return [{"id": i.id, "status": i.status, "created_at": i.created_at,
             "overall": (i.report or {}).get("overall"),
             "category_scores": (i.report or {}).get("category_scores")} for i in rows]

@app.get("/interviews/{iid}")
def get_interview(iid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return view(owned(db, iid, user))

@app.post("/interviews/{iid}/answer")
def answer(iid: int, req: AnswerReq, user: User = Depends(limit), db: Session = Depends(get_db)):
    iv = owned(db, iid, user)
    if iv.status != "active":
        raise HTTPException(409, "Interview already complete")
    resume = db.get(Resume, iv.resume_id)
    turn = next(t for t in iv.turns if t.answer is None)
    turn.answer = req.answer
    turn.evaluation = pipeline.evaluate(turn.question, turn.category, req.answer, resume.analysis)
    if len(iv.turns) >= iv.total:
        iv.report = pipeline.build_report(iv.turns, resume.analysis)
        iv.status = "complete"
    else:  # adaptive: next question depends on how this answer was scored
        cat, probe = pipeline.plan_next(iv.turns)
        query = f"{turn.question} {turn.answer}" if probe else cat + " " + " ".join(resume.analysis["weaknesses"])
        q = pipeline.next_question(resume.analysis, [t.question for t in iv.turns],
                                   rag.retrieve(db, resume.id, query), cat, probe)
        iv.turns.append(Turn(idx=len(iv.turns), category=cat, question=q, is_followup=probe is not None))
    db.commit()
    return {**view(iv), "last_evaluation": turn.evaluation}

@app.exception_handler(ValidationError)
@app.exception_handler(RuntimeError)
async def ai_failure(request, exc):  # LLM output failed schema validation or JSON parsing twice
    return JSONResponse({"detail": "The AI service returned an unusable response. Please retry."}, status_code=502)

@app.get("/metrics")
def metrics(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(LLMCall)).all()
    if not rows:
        return {"calls": 0}
    n = len(rows)
    cost = sum(r.input_tokens * llm.PRICE_IN + r.output_tokens * llm.PRICE_OUT for r in rows) / 1e6
    done = db.scalar(select(func.count()).select_from(Interview).where(Interview.status == "complete")) or 0
    return {"calls": n, "avg_latency_ms": round(mean(r.latency_ms for r in rows)),
            "avg_tokens": round(mean(r.input_tokens + r.output_tokens for r in rows)),
            "failure_rate": round(sum(not r.ok for r in rows) / n, 4),
            "retry_rate": round(sum(r.retries > 0 for r in rows) / n, 4),
            "est_total_cost_usd": round(cost, 4),
            "est_cost_per_interview_usd": round(cost / done, 4) if done else None}
