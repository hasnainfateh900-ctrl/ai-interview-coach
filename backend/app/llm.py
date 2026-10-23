"""Single choke point for LLM calls: JSON-only output, retry on malformed JSON, per-call metrics.
With no ANTHROPIC_API_KEY set, a deterministic mock is used (and nothing is logged)."""
import json, os, time

MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
# $ per million tokens: set these to your model's real pricing; cost figures are estimates.
PRICE_IN = float(os.getenv("PRICE_IN_PER_MTOK", "3"))
PRICE_OUT = float(os.getenv("PRICE_OUT_PER_MTOK", "15"))

def parse_json(text: str) -> dict:
    """Decode the first complete JSON object; tolerates prose, code fences and stray braces after it."""
    i = text.find("{")
    if i < 0:
        raise ValueError("no JSON object in response")
    return json.JSONDecoder().raw_decode(text[i:])[0]

def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def _log(**kw):
    try:
        from .db import SessionLocal
        from .models import LLMCall
        with SessionLocal() as db:
            db.add(LLMCall(model=MODEL, **kw)); db.commit()
    except Exception:
        pass  # observability must never break a request

def complete_json(system: str, user: str, mock) -> dict:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return mock()
    client, t0, tin, tout = _client(), time.monotonic(), 0, 0
    ms = lambda: int((time.monotonic() - t0) * 1000)
    for attempt in range(2):
        try:
            r = client.messages.create(
                model=MODEL, max_tokens=1500,
                system=system + "\nRespond with ONLY a valid JSON object.",
                messages=[{"role": "user", "content": user}],
            )
        except Exception:
            _log(latency_ms=ms(), input_tokens=tin, output_tokens=tout, retries=attempt, ok=False)
            raise
        tin += r.usage.input_tokens; tout += r.usage.output_tokens
        try:
            out = parse_json(r.content[0].text)
        except ValueError:
            continue
        _log(latency_ms=ms(), input_tokens=tin, output_tokens=tout, retries=attempt, ok=True)
        return out
    _log(latency_ms=ms(), input_tokens=tin, output_tokens=tout, retries=1, ok=False)
    raise RuntimeError("LLM returned invalid JSON twice")
