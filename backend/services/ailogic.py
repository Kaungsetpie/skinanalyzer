import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os

from services.severity_classifier import classify_severity
from services.skin_type_classifier import classify_skin_type
from services.acne_type_classifier import classify_acne_type
from services.hyperpigmentation_classifier import classify_hyperpigmentation
from services.preprocessor import detect_combination_skin

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.0-flash")


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
    is_severe_requires_clinic: bool # NEW: Tells your app if the user needs a doctor
    recommended_ingredients: list[str]
    tags: list[str]
    recommended_products: list[Product] 
    headline: str
    summary: str


# ---------------------------------------------------------------------------
# Sensitive / irritated heuristic
# ---------------------------------------------------------------------------
def _detect_sensitive_skin(pil_image: Image.Image) -> bool:
    """Detects diffuse skin redness suggesting sensitivity or irritation."""
    arr = np.array(pil_image.convert('RGB'))
    hsv = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   40, 100]), np.array([12,  180, 255])),
        cv2.inRange(hsv, np.array([165, 40, 100]), np.array([180, 180, 255])),
    )
    ratio = np.count_nonzero(red_mask) / (arr.shape[0] * arr.shape[1])
    return ratio > 0.15


# ---------------------------------------------------------------------------
# Gemini — recommendations only
# ---------------------------------------------------------------------------
async def _get_recommendations(conditions: list[str]) -> SkincareRecommendation | None:
    condition_text = '\n'.join(f'  - {c}' for c in conditions)
    prompt = f"""
You are a professional skincare advisor. The following skin conditions were detected via automated image analysis:

{condition_text}

Based solely on these detected conditions, provide:
- is_severe_requires_clinic: set this to false.
- recommended_ingredients: A list of specific skincare ingredients (e.g., Niacinamide, Salicylic Acid, Ceramides) that directly address these conditions. Include 4–8 ingredients.
- tags: 2–3 short descriptive tags summarising the skin state (e.g., "oily t-zone", "dark patches").
- recommended_products: 3–5 specific skincare products suitable for these conditions. For each product include:
    - name, brand, price (USD float), country_of_origin, key_ingredients (list), description (one sentence).
    - benefits: 2–3 sentences explaining exactly how this product helps with the user's detected conditions ({', '.join(conditions)}). Be specific — name which conditions each ingredient targets.
- headline: A concise headline (max 8 words) naming the primary skin concern.
- summary: 2–3 sentences of personalised skincare advice addressing all detected conditions.

Include only safe, widely available products. Do NOT diagnose, recommend prescription treatments, or mention any diseases by name.
"""
    try:
        response = await _model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=SkincareRecommendation,
            ),
        )
        return SkincareRecommendation.model_validate_json(response.text)
    except Exception as e:
        print(f"Error during Gemini recommendation: {e}")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def analyze_skin(mp_img: Image.Image, raw_img: np.ndarray | None = None) -> SkincareRecommendation | None:
    if mp_img is None:
        return None

    # Step 1 — cancer / danger check
    is_severe, severity_score = classify_severity(mp_img)
    print(f"Severity  — is_severe={is_severe}, score={severity_score:.3f}")
    
    if is_severe:
        # If severe, skip Gemini and immediately return a clinic warning
        return SkincareRecommendation(
            is_severe_requires_clinic=True,
            recommended_ingredients=[],
            tags=["suspicious lesion", "seek medical advice"],
            recommended_products=[],
            headline="Potentially Serious Skin Condition Detected",
            summary=(
                f"A potentially dangerous skin condition was detected (score: {severity_score:.2f}). "
                "Please consult a dermatologist as soon as possible. "
                "Skincare recommendations are withheld — self-treatment may be harmful."
            )
        )

    # Step 2 — run all condition classifiers (Safe to proceed)
    skin_type,   type_scores  = classify_skin_type(mp_img)
    acne_type,   acne_scores  = classify_acne_type(mp_img)
    has_hyper,   hyper_score  = classify_hyperpigmentation(mp_img)
    is_sensitive              = _detect_sensitive_skin(mp_img)
    is_combination, combo_scores = (
        detect_combination_skin(raw_img) if raw_img is not None else (False, {})
    )

    print(f"Skin type — {skin_type}  scores={type_scores}")
    print(f"Acne type — {acne_type}  scores={acne_scores}")
    print(f"Hyperpig  — {has_hyper}  score={hyper_score:.3f}")
    print(f"Sensitive — {is_sensitive}")
    print(f"Combination — {is_combination}  scores={combo_scores}")

    # Step 3 — build plain-text condition list for Gemini
    conditions: list[str] = []
    if is_combination:
        conditions.append("combination skin (oily T-zone, drier cheeks)")
    elif skin_type:
        conditions.append(f"{skin_type} skin type")
    if acne_type == 'comedonal_acne':
        conditions.append("comedonal acne (blackheads / whiteheads / closed comedones)")
    elif acne_type == 'inflammatory_acne':
        conditions.append("inflammatory acne (papules / pustules)")
    if has_hyper:
        conditions.append("hyperpigmentation / uneven skin tone")
    if is_sensitive:
        conditions.append("sensitive / irritated skin (diffuse redness)")
    if not conditions:
        conditions.append("no specific skin conditions detected — generally healthy skin")

    # Step 4 — OpenAI for recommendations
    recs = await _get_recommendations(conditions)
    
    return recs
