import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Zeabur injects DATABASE_URL like: postgres://user:password@host:port/dbname
# SQLAlchemy 1.4+ requires postgresql:// instead of postgres://
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ebm_logs.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

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
