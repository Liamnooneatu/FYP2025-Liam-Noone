from .database import Base
from sqlalchemy import Column, Integer, String, Float

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String)
    text_detected = Column(String)
    image_path = Column(String)
    word_doc_path = Column(String)
    lpips_distance = Column(Float)
