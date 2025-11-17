from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Detection
from schemas import DetectionCreate, DetectionResponse
from typing import List

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get SQL session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {"status": "FastAPI backend running"}


@app.post("/api/detections", response_model=DetectionResponse)
def create_detection(data: DetectionCreate, db: Session = Depends(get_db)):
    detection = Detection(
        timestamp=data.timestamp,
        text_detected=data.text_detected,
        image_path=data.image_path,
        word_doc_path=data.word_doc_path,
        lpips_distance=data.lpips_distance
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


@app.get("/api/detections", response_model=List[DetectionResponse])
def get_detections(db: Session = Depends(get_db)):
    return db.query(Detection).order_by(Detection.id.desc()).all()
