from fastapi import FastAPI
from fastapi import File, UploadFile
from Validators import FileUpload
from fastapi.middleware.cors import CORSMiddleware
from services.ailogic import analyze_skin
from services.preprocessor import preprocess_image
import numpy as np
import cv2


app = FastAPI()

origins = ["*"]  # Allow all origins for development; adjust in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/")
def main():
    return {"message": "Skinalyzer API is running"}


@app.post("/analysis/upload")
async def upload_file(file: UploadFile = File(...)):
    bytes = await file.read()

    # convert bytes to numpy array
    np_array = np.frombuffer(bytes, np.uint8)
    img_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    # mediapipe expects RGB format, convert from BGR to RGB
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cropped_face = preprocess_image(img_rgb)
    response = await analyze_skin(cropped_face)
    if response:
        print(f"Skin analysis response: {response}")
        return {
            "status": "success",
            "data": response.dict()
        }
    else:
        return {"error": "Skin analysis failed"}
