from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import shutil

# folder to save uploaded Word docs
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI(title="Camera Change Detection API")

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload_word/")
async def upload_word(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/uploaded_files/")
def list_uploaded_files():
    files = os.listdir(UPLOAD_FOLDER)
    return {"files": files}

@app.get("/download_word/{filename}")
def download_word(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=filename
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")

# endpoint
@app.get("/")
def root():
    return {"status": "FastAPI backend running"}
