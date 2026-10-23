import os
os.environ["ENV"] = "dev"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ.pop("ANTHROPIC_API_KEY", None)  # forces deterministic mock LLM
if os.path.exists("test.db"):
    os.remove("test.db")
