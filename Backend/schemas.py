from pydantic import BaseModel

class DetectionCreate(BaseModel):
    timestamp: str
    text_detected: str
    image_path: str
    word_doc_path: str
    lpips_distance: float

class DetectionResponse(DetectionCreate):
    id: int

    class Config:
        orm_mode = True
