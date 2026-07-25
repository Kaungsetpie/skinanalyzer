from db import AnalysisResult, get_db
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from Validators import FileUpload
from fastapi.middleware.cors import CORSMiddleware
from services.ailogic import analyze_skin
from services.preprocessor import preprocess_image
import json
import numpy as np
import cv2
import uuid
import datetime
from sqlalchemy.orm import Session

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
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    print(f"Received file: {file.filename}, content type: {file.content_type}")
    try:
        bytes = await file.read()

        # convert bytes to numpy array
        np_array = np.frombuffer(bytes, np.uint8)
        img_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        cropped_face = preprocess_image(img_np)
        if cropped_face is None:
            raise HTTPException(status_code=400, detail="No face detected in the image. Please upload a clear face photo.")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {str(e)}")

    try:
        response = await analyze_skin(cropped_face, img_np)
    
        print(f"AI response: {response}")
    
        id = str(uuid.uuid4())
        created_at = datetime.datetime.now()
    
        if hasattr(response,"model_dump"):
            data = response.model_dump()
        elif hasattr(response,"dict"):
            data = response.dict()
        else:
            data = response.__dict__
    
        new_analysis = AnalysisResult(
            id=id,
            headline=data.get("headline", ""),  
            created_at=created_at,
            dermatologist_required=data.get(
                "dermatologist_recommendation", False),
                recommended_products=data.get("recommended_products", []),
            ingredients=data.get("recommended_ingredients", []),
            analysis_summary=data.get("summary", ""),
            tags=data.get("tags", []),
            full_analysis=data
        )
        db.add(new_analysis)
        db.commit()
        db.refresh(new_analysis)
        print(f"Analysis result saved to database with ID: {new_analysis.id}")
        return {
            "statusCode": 200,
            "message": "Skin analysis completed successfully",
            "id": new_analysis.id,
        }
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error decoding JSON response: {str(e)}")
    except Exception as e:
        status = getattr(e, 'status_code', None) or getattr(e, 'code', None)
        if status == 503 or '503' in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        print(f"Error during skin analysis: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Error during skin analysis: {str(e)}")
    

@app.get("/analysis/history")
def get_all_analysis_results(db: Session = Depends(get_db)):
    analyses = db.query(AnalysisResult).all()
    return analyses

@app.get("/analysis/{analysis_id}")
def get_analysis_result(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    print(analysis)
    if(analysis):
        return analysis 

