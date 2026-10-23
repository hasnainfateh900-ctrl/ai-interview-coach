import os, datetime as dt
import bcrypt, jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from .db import get_db
from .models import User

SECRET = os.getenv("JWT_SECRET")
if not SECRET:
    if os.getenv("ENV") != "dev":
        raise RuntimeError("Set JWT_SECRET (or ENV=dev for local development)")
    SECRET = "dev-only-secret"
bearer = HTTPBearer()

def hash_pw(p: str) -> str:
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()

def check_pw(p: str, h: str) -> bool:
    return bcrypt.checkpw(p.encode(), h.encode())

def make_token(uid: int) -> str:
    exp = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=12)
    return jwt.encode({"sub": str(uid), "exp": exp}, SECRET, "HS256")

def current_user(cred=Depends(bearer), db=Depends(get_db)) -> User:
    try:
        uid = int(jwt.decode(cred.credentials, SECRET, ["HS256"])["sub"])
    except Exception:
        raise HTTPException(401, "Invalid or expired token")
    user = db.get(User, uid)
    if not user:
        raise HTTPException(401, "Unknown user")
    return user

# Per-user sliding-window limiter on LLM-backed routes (in-memory: per process; use a shared store when scaling out).
import time
from collections import defaultdict, deque
_hits = defaultdict(deque)
LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))

def limit(user: User = Depends(current_user)) -> User:
    q, now = _hits[user.id], time.monotonic()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= LIMIT:
        raise HTTPException(429, "Too many requests, slow down")
    q.append(now)
    return user
