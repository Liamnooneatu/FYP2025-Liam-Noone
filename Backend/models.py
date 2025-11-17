from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from database import Base
from datetime import datetime

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    text_detected = Column(Text)
    image_path = Column(String(255))
    word_doc_path = Column(String(255))
    lpips_distance = Column(Float)
