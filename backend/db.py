import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Boolean, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

database_url = 'sqlite:///analysis.db'
engine = create_engine(database_url, connect_args={"check_same_thread": False})
session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AnalysisResult(Base):
    __tablename__ = 'analysis'
    id = Column(String, primary_key=True, index=True)
    headline = Column(String)
    created_at = Column(DateTime)
    dermatologist_required = Column(Boolean)
    ingredients = Column(JSON)
    recommended_products = Column(JSON)
    analysis_summary = Column(String)
    full_analysis = Column(JSON)
    tags = Column(JSON)

    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, \
                dermatologist_required={self.dermatologist_required})>"


Base.metadata.create_all(bind=engine)


def get_db():
    db = session_local()
    try:
        yield db
    finally:
        db.close()
