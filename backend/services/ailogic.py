from PIL import Image
from pydantic import BaseModel
import numpy as np
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMIAI_API_KEY"))


class SkinAnalysis(BaseModel):
    whiteheads: int
    blackheads: int
    imflammatory_acne: int
    suspicious_lesions: int
    oiliness: int
    dryness: int
    enlarged_pores: int
    hyperpigmentation: int
    wrinkles: int
    dullness: int
    dermatologist_recommendation: bool
    recommended_ingredients: list[str]
    summary: str


async def analyze_skin(mp_img):
    prompt = """
        Perform a clinical-style skin analysis. 
        For each category, provide a severity score from 0 (none) to 100 (severe).
        Determine if the user should see a dermatologist based on inflammatory acne or suspicious lesions.
        List specific skincare ingredients (e.g., Salicylic Acid, Niacinamide) that help.
        """
    try:
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                prompt,
                mp_img
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": SkinAnalysis
                }
        )
        return response.parsed
    except Exception as e:
        print(f"Error during skin analysis: {e}")
        return None
