import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

from services.severity_classifier import classify_severity
from services.skin_type_classifier import classify_skin_type
from services.acne_type_classifier import classify_acne_type
from services.hyperpigmentation_classifier import classify_hyperpigmentation
from services.preprocessor import detect_combination_skin

load_dotenv()
_gemini_api_key = os.getenv("GEMINI_API_KEY")
_gemini_client = genai.Client(api_key=_gemini_api_key) if _gemini_api_key else None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class Product(BaseModel):
    name: str
    brand: str
    price: float
    country_of_origin: str
    key_ingredients: list[str]
    benefits: str
    description: str

class SkincareRecommendation(BaseModel):
    is_severe_requires_clinic: bool
    recommended_ingredients: list[str]
    tags: list[str]
    recommended_products: list[Product]
    headline: str
    summary: str


# ---------------------------------------------------------------------------
# Sensitive skin heuristic
# ---------------------------------------------------------------------------
def _detect_sensitive_skin(pil_image: Image.Image) -> bool:
    arr = np.array(pil_image.convert('RGB'))
    hsv = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   40, 100]), np.array([12,  180, 255])),
        cv2.inRange(hsv, np.array([165, 40, 100]), np.array([180, 180, 255])),
    )
    ratio = np.count_nonzero(red_mask) / (arr.shape[0] * arr.shape[1])
    return ratio > 0.15


# ---------------------------------------------------------------------------
# Step 1 — run all classifiers, return a plain dict
# ---------------------------------------------------------------------------
def classify_conditions(mp_img: Image.Image, raw_img: np.ndarray | None = None) -> dict:
    is_severe, severity_score = classify_severity(mp_img)
    print(f"Severity  — is_severe={is_severe}, score={severity_score:.3f}")

    if is_severe:
        return {"is_severe": True, "severity_score": round(float(severity_score), 3)}

    skin_type,  type_scores  = classify_skin_type(mp_img)
    acne_type,  acne_scores  = classify_acne_type(mp_img)
    has_hyper,  hyper_score  = classify_hyperpigmentation(mp_img)
    is_sensitive             = _detect_sensitive_skin(mp_img)
    is_combination, _        = detect_combination_skin(raw_img) if raw_img is not None else (False, {})

    print(f"Skin type — {skin_type}  scores={type_scores}")
    print(f"Acne type — {acne_type}  scores={acne_scores}")
    print(f"Hyperpig  — {has_hyper}  score={hyper_score:.3f}")
    print(f"Sensitive — {is_sensitive}")
    print(f"Combination — {is_combination}")

    # Cast everything to native Python types so SQLAlchemy's JSON column can serialize them
    return {
        "is_severe": False,
        "skin_type": str(skin_type) if skin_type is not None else None,
        "acne_type": str(acne_type) if acne_type is not None else None,
        "has_hyperpigmentation": bool(has_hyper),
        "is_sensitive": bool(is_sensitive),
        "is_combination": bool(is_combination),
    }


def _build_conditions_list(conditions: dict) -> list[str]:
    conds: list[str] = []
    if conditions.get("is_combination"):
        conds.append("combination skin (oily T-zone, drier cheeks)")
    elif conditions.get("skin_type"):
        conds.append(f"{conditions['skin_type']} skin type")
    acne = conditions.get("acne_type")
    if acne == "comedonal_acne":
        conds.append("comedonal acne (blackheads / whiteheads / closed comedones)")
    elif acne == "inflammatory_acne":
        conds.append("inflammatory acne (papules / pustules)")
    if conditions.get("has_hyperpigmentation"):
        conds.append("hyperpigmentation / uneven skin tone")
    if conditions.get("is_sensitive"):
        conds.append("sensitive / irritated skin (diffuse redness)")
    if not conds:
        conds.append("no specific skin conditions detected — generally healthy skin")
    return conds


# ---------------------------------------------------------------------------
# Gemini — multi-model fallback
# ---------------------------------------------------------------------------
_GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
]

async def _call_gemini(prompt: str) -> SkincareRecommendation | None:
    if _gemini_client is None:
        print("GEMINI_API_KEY is not configured — using classifier-only recommendations.")
        return None

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SkincareRecommendation,
    )
    for model in _GEMINI_MODELS:
        try:
            response = await _gemini_client.aio.models.generate_content(
                model=model, contents=prompt, config=config,
            )
            print(f"Gemini response from model: {model}")
            return SkincareRecommendation.model_validate_json(response.text)
        except Exception as e:
            err = str(e)
            if (
                "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower()
                or "404" in err or "NOT_FOUND" in err
            ):
                print(f"Skipping {model} ({err[:80]}…), trying next…")
                continue
            print(f"Error during Gemini recommendation ({model}): {e}")
            return None
    print("All Gemini models unavailable — returning classifier-only result.")
    return None


# ---------------------------------------------------------------------------
# Step 2 — call Gemini for recommendations (called separately by main.py)
# ---------------------------------------------------------------------------
async def get_recommendations(conditions: dict, budget: float) -> SkincareRecommendation:
    condition_list = _build_conditions_list(conditions)
    condition_text = '\n'.join(f'  - {c}' for c in condition_list)

    prompt = f"""
You are a professional skincare advisor. The user's skin analysis detected the following conditions:

{condition_text}

The user's budget is ${budget:.0f} USD.

Your task: recommend a complete, personalised skincare routine that directly targets these specific conditions. Every product and ingredient must be chosen because it addresses one or more of the detected conditions above — not as generic skincare.

Return JSON with:
- is_severe_requires_clinic: false
- recommended_ingredients: 4–8 active ingredients that directly address these exact conditions (explain in your head why each one, but only list the names).
- tags: 2–3 short labels describing the user's skin state (e.g. "oily t-zone", "comedonal acne", "dark spots").
- recommended_products: exactly 12 real, purchasable skincare products spread across price tiers:
    * At least 5 products priced AT OR BELOW ${budget:.0f} USD — these are the priority picks
    * The remaining products at mid-range ($30–$80) or premium ($80+) price points
  For each product include: name, brand, price (USD float), country_of_origin, key_ingredients (list of the active ingredients in that product), description (one sentence stating what the product is), benefits (2–3 sentences explaining specifically how this product helps the detected conditions).
- headline: ≤8 words naming the primary skin concern (e.g. "Oily Skin with Comedonal Acne").
- summary: 2–3 sentences of personalised advice addressing these conditions and how the recommended routine helps.

Use only real, widely available products. Do NOT recommend prescription treatments or diagnose medical conditions.
"""

    recs = await _call_gemini(prompt)
    if recs is not None:
        return recs

    # Gemini unavailable — return classifier-only fallback
    tags = []
    if conditions.get("is_combination"):
        tags.append("combination skin")
    elif conditions.get("skin_type"):
        tags.append(str(conditions["skin_type"]))
    acne = conditions.get("acne_type")
    if acne and acne != "no_acne":
        tags.append(str(acne).replace("_", " "))
    if conditions.get("has_hyperpigmentation"):
        tags.append("hyperpigmentation")
    if conditions.get("is_sensitive"):
        tags.append("sensitive skin")

    condition_lines = "\n".join(f"• {c}" for c in condition_list)
    return SkincareRecommendation(
        is_severe_requires_clinic=False,
        recommended_ingredients=[],
        tags=tags,
        recommended_products=[],
        headline="Skin Analysis Complete",
        summary=(
            f"Detected conditions:\n{condition_lines}\n\n"
            "Product recommendations are temporarily unavailable (AI quota exceeded). "
            "Try again later for personalised product suggestions."
        ),
    )
