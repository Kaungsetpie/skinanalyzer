from db import AnalysisResult, get_db
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.ailogic import classify_conditions, get_recommendations
from services.preprocessor import preprocess_image
from pydantic import BaseModel
from sqlalchemy.orm.attributes import flag_modified
import numpy as np
import cv2
import uuid
import datetime
from sqlalchemy.orm import Session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/")
def main():
    return {"message": "Skinalyzer API is running"}


# ---------------------------------------------------------------------------
# Step 1 — upload image, run classifiers, return skin conditions
# ---------------------------------------------------------------------------
@app.post("/analysis/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    print(f"Received file: {file.filename}, content type: {file.content_type}")
    try:
        bytes_data = await file.read()
        np_array = np.frombuffer(bytes_data, np.uint8)
        img_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        cropped_face = preprocess_image(img_np)
        if cropped_face is None:
            raise HTTPException(status_code=400, detail="No face detected. Please upload a clear face photo.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    try:
        conditions = classify_conditions(cropped_face, img_np)
        print("="*60)
        print(" 🧪 LIVE DIAGNOSTIC ANALYSIS RESULT:")
        print(f"    • Severity           : {conditions.get('is_severe')} (Score: {conditions.get('severity_score', 0)})")
        print(f"    • Skin Type          : {conditions.get('skin_type')}")
        print(f"    • Acne Condition     : {conditions.get('acne_type')}")
        print(f"    • Hyperpigmentation  : {conditions.get('has_hyperpigmentation')}")
        print("="*60)

        analysis_id = str(uuid.uuid4())
        new_analysis = AnalysisResult(
            id=analysis_id,
            created_at=datetime.datetime.now(),
            headline="",
            dermatologist_required=conditions.get("is_severe", False),
            ingredients=[],
            recommended_products=[],
            analysis_summary="",
            tags=[],
            full_analysis={"conditions": conditions},
        )
        db.add(new_analysis)
        db.commit()

        return {"analysisId": analysis_id, "conditions": conditions}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error during skin analysis: {str(e)}")


# ---------------------------------------------------------------------------
# Step 2 — fetch Gemini recommendations for a given budget
# ---------------------------------------------------------------------------
class RecommendationRequest(BaseModel):
    budget: float = 50.0


@app.post("/analysis/recommendations/{analysis_id}")
async def create_recommendations(
    analysis_id: str,
    req: RecommendationRequest,
    db: Session = Depends(get_db),
):
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    conditions = (analysis.full_analysis or {}).get("conditions", {})

    try:
        recs = await get_recommendations(conditions, req.budget)

        analysis.headline = recs.headline
        analysis.analysis_summary = recs.summary
        analysis.tags = recs.tags
        analysis.ingredients = recs.recommended_ingredients
        analysis.recommended_products = [p.model_dump() for p in recs.recommended_products]
        analysis.dermatologist_required = recs.is_severe_requires_clinic

        existing = dict(analysis.full_analysis or {})
        existing.update(recs.model_dump())
        analysis.full_analysis = existing
        flag_modified(analysis, "full_analysis")

        db.commit()
        db.refresh(analysis)

        return {"statusCode": 200}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error fetching recommendations: {str(e)}")


# ---------------------------------------------------------------------------
# History & detail
# ---------------------------------------------------------------------------
@app.get("/analysis/history")
def get_all_analysis_results(db: Session = Depends(get_db)):
    return db.query(AnalysisResult).all()


@app.get("/analysis/{analysis_id}")
def get_analysis_result(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    return analysis
