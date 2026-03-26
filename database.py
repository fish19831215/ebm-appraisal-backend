import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Zeabur injects POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, etc.
# Check these fallback environment variables first
pg_user = os.getenv("POSTGRES_USER")
pg_pass = os.getenv("POSTGRES_PASSWORD")
pg_host = os.getenv("POSTGRES_HOST")
pg_port = os.getenv("POSTGRES_PORT", "5432")
pg_db = os.getenv("POSTGRES_DB", "postgres")

if pg_user and pg_host:
    DATABASE_URL = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
else:
    # Use standard DATABASE_URL if available, else fallback to sqlite
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ebm_logs.db")
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Debug printing for Zeabur logs
print("===============[ DB Connection string (masked) ]===============")
safe_url = DATABASE_URL
if "@" in safe_url:
    safe_url = safe_url.split("://")[0] + "://[USER]:[PASS]@" + safe_url.split("@")[1]
print(f"Connecting to: {safe_url}")
print("=============================================================")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action_type = Column(String(50), index=True) # e.g. "CHAT", "EXTRACT_PICO", "SEARCH", "REPORT"
    details = Column(Text) # JSON string or specific query text

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
