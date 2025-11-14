import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import create_document, get_documents

app = FastAPI(title="AI Crop Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SoilData(BaseModel):
    ph: float = Field(..., ge=0, le=14)
    moisture: float = Field(..., ge=0, le=100, description="Percentage")
    nitrogen: float = Field(..., ge=0, description="ppm or kg/ha equivalent")
    phosphorus: float = Field(..., ge=0, description="ppm or kg/ha equivalent")
    potassium: float = Field(..., ge=0, description="ppm or kg/ha equivalent")


class WeatherData(BaseModel):
    rainfall_mm: float = Field(..., ge=0)
    temperature_c: float


class MarketData(BaseModel):
    demand_index: float = Field(..., ge=0, le=1, description="0-1 scale")
    price_index: float = Field(..., ge=0, le=1, description="0-1 scale")


class RecommendationRequest(BaseModel):
    location: str
    soil: SoilData
    weather: WeatherData
    previous_crops: List[str] = []
    market: Optional[MarketData] = None
    preferred_language: Optional[str] = "en"


class CropRecommendation(BaseModel):
    crop: str
    score: float
    expected_yield_tpha: float
    profit_index: float
    sustainability_score: float
    notes: str


class RecommendationResponse(BaseModel):
    recommendations: List[CropRecommendation]
    generated_at: datetime


# Simple rule-based engine for prototype
CROP_RULES = [
    {
        "name": "Wheat",
        "ph": (6.0, 7.5),
        "moisture": (40, 70),
        "npk_ratio": (2, 1, 1),
        "base_yield": 4.0,
    },
    {
        "name": "Rice",
        "ph": (5.5, 7.0),
        "moisture": (60, 90),
        "npk_ratio": (1, 1, 1),
        "base_yield": 5.0,
    },
    {
        "name": "Maize",
        "ph": (5.5, 7.0),
        "moisture": (40, 70),
        "npk_ratio": (2, 1, 2),
        "base_yield": 6.0,
    },
    {
        "name": "Cotton",
        "ph": (5.8, 8.0),
        "moisture": (30, 60),
        "npk_ratio": (1, 1, 1),
        "base_yield": 2.0,
    },
    {
        "name": "Pulses",
        "ph": (6.0, 7.5),
        "moisture": (30, 60),
        "npk_ratio": (1, 1, 1),
        "base_yield": 1.2,
    },
]


def rotation_penalty(previous_crops: List[str], candidate: str) -> float:
    if any(candidate.lower() in c.lower() for c in previous_crops[-2:]):
        return -0.2  # discourage monocropping
    # Encourage legume rotation after cereals
    if candidate.lower() in ["pulses"] and any(c in ["wheat", "rice", "maize"] for c in [p.lower() for p in previous_crops]):
        return 0.1
    return 0.0


def market_boost(market: Optional[MarketData]) -> float:
    if not market:
        return 0.0
    # Simple average of demand and price
    return 0.3 * ((market.demand_index + market.price_index) / 2)


def compute_recommendations(req: RecommendationRequest) -> List[CropRecommendation]:
    results: List[CropRecommendation] = []

    for rule in CROP_RULES:
        # pH suitability
        ph_low, ph_high = rule["ph"]
        ph_score = 1.0 if ph_low <= req.soil.ph <= ph_high else max(0.0, 1 - (min(abs(req.soil.ph - ph_low), abs(req.soil.ph - ph_high)) / 2))

        # moisture suitability
        m_low, m_high = rule["moisture"]
        if m_low <= req.soil.moisture <= m_high:
            moisture_score = 1.0
        else:
            dist = min(abs(req.soil.moisture - m_low), abs(req.soil.moisture - m_high))
            moisture_score = max(0.0, 1 - dist / 50)

        # nutrient simple heuristic based on N:P:K proximity
        n, p, k = req.soil.nitrogen, req.soil.phosphorus, req.soil.potassium
        rn, rp, rk = rule["npk_ratio"]
        # Normalize to ratios
        total = max(n + p + k, 1)
        n_ratio, p_ratio, k_ratio = n / total, p / total, k / total
        r_total = rn + rp + rk
        rn_ratio, rp_ratio, rk_ratio = rn / r_total, rp / r_total, rk / r_total
        npk_score = max(0.0, 1 - (abs(n_ratio - rn_ratio) + abs(p_ratio - rp_ratio) + abs(k_ratio - rk_ratio)) / 1.5)

        # temperature influence (broad comfort 18-35C)
        temp = req.weather.temperature_c
        if 18 <= temp <= 35:
            temp_score = 1.0
        else:
            temp_score = max(0.0, 1 - (abs(temp - (26.5)) / 20))

        # rainfall coarse adjustment
        rain = req.weather.rainfall_mm
        rain_score = max(0.0, min(1.0, rain / 800))  # scale up to good at ~800mm

        base_score = 0.35 * ph_score + 0.25 * moisture_score + 0.15 * npk_score + 0.15 * temp_score + 0.10 * rain_score
        rot_adj = rotation_penalty(req.previous_crops, rule["name"]) \
            + market_boost(req.market)
        final_score = max(0.0, min(1.2, base_score + rot_adj))

        expected_yield = rule["base_yield"] * (0.7 + 0.6 * final_score)  # 0.7x to 1.34x base
        profit_index = max(0.0, min(1.0, 0.5 * final_score + (req.market.price_index if req.market else 0.3)))
        sustainability = max(0.0, min(1.0, 0.6 * (1 - (1 if rot_adj < 0 else 0) * 0.2) + 0.4 * (1 - abs(req.soil.ph - 7) / 7)))

        notes = []
        if req.soil.ph < ph_low:
            notes.append("Soil slightly acidic; consider liming")
        if req.soil.ph > ph_high:
            notes.append("Soil slightly alkaline; consider gypsum/organic matter")
        if req.soil.moisture < m_low:
            notes.append("Low soil moisture; improve irrigation/mulching")
        if req.soil.moisture > m_high:
            notes.append("High soil moisture; ensure drainage")

        results.append(CropRecommendation(
            crop=rule["name"],
            score=round(final_score, 3),
            expected_yield_tpha=round(expected_yield, 2),
            profit_index=round(profit_index, 3),
            sustainability_score=round(sustainability, 3),
            notes="; ".join(notes) if notes else "",
        ))

    # Sort by score descending and take top 5
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:5]


@app.get("/")
def root():
    return {"message": "AI Crop Recommendation API"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend(req: RecommendationRequest):
    recs = compute_recommendations(req)

    # Log to database
    try:
        log_id = create_document("recommendation", {
            "request": req.model_dump(),
            "response": [r.model_dump() for r in recs],
            "created_at": datetime.utcnow(),
        })
    except Exception:
        log_id = None

    return RecommendationResponse(recommendations=recs, generated_at=datetime.utcnow())


@app.get("/api/auto-data")
async def auto_data(lat: float, lon: float):
    import random
    # In production, call SoilGrids/Bhuvan/IMD APIs here
    ph = round(random.uniform(5.5, 7.8), 2)
    moisture = round(random.uniform(30, 85), 1)
    n = round(random.uniform(20, 200), 1)
    p = round(random.uniform(10, 60), 1)
    k = round(random.uniform(20, 200), 1)
    rainfall = round(random.uniform(400, 1200), 1)
    temp = round(random.uniform(18, 35), 1)
    demand = round(random.uniform(0.3, 0.9), 2)
    price = round(random.uniform(0.3, 0.9), 2)

    return {
        "location": f"{lat:.3f}, {lon:.3f}",
        "soil": {"ph": ph, "moisture": moisture, "nitrogen": n, "phosphorus": p, "potassium": k},
        "weather": {"rainfall_mm": rainfall, "temperature_c": temp},
        "market": {"demand_index": demand, "price_index": price},
    }


@app.get("/api/history")
async def history(limit: int = 10):
    try:
        docs = get_documents("recommendation", {}, limit=limit)
        # Convert ObjectId to str safely
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])
        return {"items": docs}
    except Exception as e:
        # Database might not be configured
        return {"items": [], "warning": str(e)}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
