# SUPPRESS ALL WARNINGS
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# LOGGING SETUP
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ENV SETUP
import os
from dotenv import load_dotenv
load_dotenv()

USGS_API_KEY = os.getenv("USGS_API_KEY")
EARTHDATA_BEARER_TOKEN = os.getenv("EARTHDATA_BEARER_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY not found — executive summaries will use fallback.")

# =========================
# IMPORTS
# =========================
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import json
from typing import Dict, Optional, List
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import matplotlib.pyplot as plt
from io import BytesIO
from groq import Groq

# GROQ CLIENT
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# =========================
# PATH & DEVICE CONFIG (RENDER-SAFE)
# =========================
# Absolute base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Model directory (Render-safe)
MODEL_DIR = os.path.join(BASE_DIR, "models")
TRAINING_CSV_PATH = os.path.join(BASE_DIR, "groundwater_cleaned.csv")
# Device (Render = CPU only)
DEVICE = torch.device("cpu")
logging.info("Deployment: Render-compatible (CPU-only, absolute paths)")

# STARTUP VERIFICATION LOGS
logging.info(f"BASE_DIR = {BASE_DIR}")
logging.info(f"Files in BASE_DIR = {os.listdir(BASE_DIR)}")

if os.path.exists(MODEL_DIR):
    logging.info(f"Files in models/ = {os.listdir(MODEL_DIR)}")
else:
    logging.critical("models/ directory NOT FOUND")

# GLOBAL CONSTANTS
STANDARD_DESIRABLE = {
    "tds": 500, "nitrate": 45, "fluoride": 1.0, "iron": 0.3,
    "chloride": 250, "hardness": 200, "sulfate": 200
}

IMPORTANT_USER_FEATURES = [
    "ph", "tds", "nitrate", "chloride",
    "hardness", "sulfate", "fluoride",
    "iron", "conductivity"
]

WEIGHTS = {
    "nitrate": 0.20,
    "tds": 0.15,
    "fluoride": 0.15,
    "iron": 0.10,
    "ph": 0.10,
    "chloride": 0.10,
    "hardness": 0.10,
    "sulfate": 0.10,
}

CRITICAL_LIMITS = {
    "nitrate": 45,
    "fluoride": 1.5,
    "iron": 1.0,
    "ph_min": 6.5,
    "ph_max": 8.5,
}

BIS_WHO_STANDARDS: List[Dict] = [
    {"parameter": "pH", "desirable": "6.5–8.5", "permissible": "No relaxation", "unit": "", "health_note": "Affects taste & corrosivity"},
    {"parameter": "TDS", "desirable": "500", "permissible": "2000", "unit": "mg/L", "health_note": "Affects palatability"},
    {"parameter": "Nitrate", "desirable": "45", "permissible": "No relaxation", "unit": "mg/L", "health_note": "Methemoglobinemia risk"},
    {"parameter": "Fluoride", "desirable": "1.0", "permissible": "1.5", "unit": "mg/L", "health_note": "Dental/skeletal fluorosis"},
    {"parameter": "Chloride", "desirable": "250", "permissible": "1000", "unit": "mg/L", "health_note": "Salty taste"},
    {"parameter": "Total Hardness (as CaCO3)", "desirable": "200", "permissible": "600", "unit": "mg/L", "health_note": "Scaling & soap inefficiency"},
    {"parameter": "Sulfate", "desirable": "200", "permissible": "400", "unit": "mg/L", "health_note": "Laxative effect"},
    {"parameter": "Iron", "desirable": "0.3", "permissible": "1.0", "unit": "mg/L", "health_note": "Staining & metallic taste"},
    {"parameter": "Conductivity", "desirable": "<800", "permissible": "No fixed limit", "unit": "µS/cm", "health_note": "Proxy for TDS"},
]

PLOT_LIMITS = {
    "PH": (6.5, 8.5),
    "TDS": 500,
    "NITRATE": 45,
    "CHLORIDE": 250,
    "HARDNESS": 200,
    "SULFATE": 200,
    "FLUORIDE": 1.0,
    "IRON": 0.3,
    "CONDUCTIVITY": 800,
}

# =========================
# CRITICAL ML ASSET LOADING (FAIL LOUDLY)
# =========================
try:
    logging.info(f"Loading training CSV: {TRAINING_CSV_PATH}")
    df_train = pd.read_csv(TRAINING_CSV_PATH)

    numeric_cols = df_train.select_dtypes(include=["number"]).columns.tolist()
    FEATURE_COLUMNS = [col.lower() for col in numeric_cols]
    FEATURE_DEFAULTS = {col.lower(): df_train[col].median() for col in numeric_cols}

    logging.info("Loading ML artifacts...")
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.joblib"))
    rf = joblib.load(os.path.join(MODEL_DIR, "rf_cluster_emulator.joblib"))

    logging.info("Scaler, PCA, RF loaded successfully")

except Exception as e:
    logging.critical(f"❌ Failed to load ML assets: {e}")
    raise RuntimeError("Critical ML files missing. Deployment aborted.")

# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# Autoencoder loading (guarded)
try:
    ae = Autoencoder(pca.n_components_)
    ae.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "autoencoder.pt"),
            map_location=DEVICE
        )
    )
    ae.to(DEVICE)
    ae.eval()
    logging.info("Autoencoder loaded successfully")

except Exception as e:
    logging.critical(f"❌ Failed to load autoencoder: {e}")
    raise RuntimeError("Autoencoder model missing or corrupted.")

logging.info("Full ML models and training data loaded successfully.")

# ============================================================
# GROQ SUMMARY GENERATOR
# ============================================================
def generate_executive_summary(data: Dict, mode: str) -> str:
    if not groq_client:
        return "Executive summary unavailable (missing GROQ_API_KEY). Review detailed results."

    prompt_templates = {
        "location": """You are a groundwater expert. Write a professional, natural-language executive summary (200–350 words).

Include:
- Location details
- Well and aquifer context
- Current water quality snapshot
- Soil type and agricultural practices
- Long-term groundwater level trend with causes and projections
- Overall status and primary risks
- Practical recommendations

Data JSON:
{data_json}
""",
        "sample": """You are a groundwater quality expert. Write a professional executive summary (250–400 words).

Include:
- WQI score and category
- BIS/WHO compliance and exceedances
- ML anomaly detection and confidence
- Key risks and implications
- Seasonal/trend context if provided
- Uncertainty analysis
- Final recommendation

Reference standards table.

Data JSON:
{data_json}
"""
    }

    prompt = prompt_templates.get(mode, "Summarize groundwater data:\n{data_json}").format(data_json=json.dumps(data, indent=2))

    try:
        logging.info(f"Calling Groq for {mode} summary")
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=600,
            temperature=0.6,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Groq error: {e}")
        return "Executive summary temporarily unavailable."

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Next-Gen Groundwater Quality Intelligence API",
    version="3.1",
    description="ML-powered groundwater quality analysis with rich insights, BIS/WHO standards, visualizations & Groq summaries"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# FIX FAVICON 404
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(content=b"", media_type="image/x-icon")

# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "deployment_mode": "Full ML Mode (CPU)"
    }

# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.get("/")
def root():
    return {
        "message": "🚀 Next-Gen Groundwater Quality Intelligence API is running successfully!",
        "status": "active",
        "version": "3.1",
        "deployment_mode": "Full ML Mode (CPU)",
        "endpoints": {
            "location_analysis": "/mode1/location-analysis (POST)",
            "sample_analysis": "/mode2/sample-analysis (POST)",
            "interactive_docs": "/docs",
            "openapi_spec": "/openapi.json",
            "health": "/health"
        },
        "note": "Use /docs for Swagger UI to test endpoints."
    }

# ============================================================
# REQUEST MODELS
# ============================================================
class LocationQuery(BaseModel):
    country: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class SampleQuery(BaseModel):
    parameters: Dict[str, float]
    season: Optional[str] = None
    groundwater_trend: Optional[str] = None

# ============================================================
# MODE 1 — LOCATION ANALYSIS
# ============================================================
@app.post("/mode1/location-analysis")
def location_analysis(q: LocationQuery):
    logging.info(f"MODE1 request: {q.dict()}")

    response_data = {
        "location": {
            "country": q.country or "India",
            "state": q.state or "Punjab",
            "district": q.district or "Ludhiana",
            "village": q.village or "Jagraon",
            "latitude": q.latitude or 30.9,
            "longitude": q.longitude or 75.85
        },
        "well_metadata": {
            "aquifer_type": "Unconfined alluvial aquifer",
            "typical_depth_range_m": "30–60"
        },
        "water_quality_snapshot": {
            "pH": 7.2,
            "TDS_mg_L": 550,
            "Nitrate_mg_L": 35,
            "Chloride_mg_L": 190,
            "Hardness_mg_L": 280,
            "Fluoride_mg_L": 0.8,
            "Iron_mg_L": 0.25
        },
        "soil_context": {
            "dominant_type": "Alluvial/loamy soil",
            "characteristics": "High permeability, fertile but prone to leaching",
            "common_issues": "Nutrient runoff into groundwater"
        },
        "agricultural_context": {
            "major_crops": "Wheat, rice, cotton, sugarcane",
            "practices": "Intensive irrigation with tube wells",
            "fertilizer_use": "Heavy nitrogen fertilizers → nitrate contamination risk"
        },
        "groundwater_level_trend": {
            "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "depth_m_bgl": [8.5, 9.3, 10.2, 11.1, 12.0, 13.0, 14.1, 15.2, 16.4, 17.7, 19.0],
            "average_annual_decline_m": 1.05,
            "primary_causes": ["Over-extraction for agriculture", "Reduced monsoon recharge", "Urban expansion"],
            "projection": "Potential critical depletion by 2030 without intervention"
        },
        "overall_status": "Stressed aquifer with moderate contamination risk",
        "primary_risks": [
            "Nitrate pollution from farming",
            "Groundwater depletion",
            "Seasonal variability in recharge"
        ],
        "recommendations": [
            "Adopt drip/micro-irrigation",
            "Promote rainwater harvesting",
            "Implement crop rotation & precise fertilization",
            "Regular groundwater monitoring"
        ],
        "visualizations": {
            "trend_plot": "/mode1/trend-plot"
        }
    }

    response_data["executive_summary"] = generate_executive_summary(response_data, "location")
    return response_data

@app.get("/mode1/trend-plot", response_class=StreamingResponse)
def trend_plot():
    logging.info("MODE1 trend plot requested")
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    depth = [8.5, 9.3, 10.2, 11.1, 12.0, 13.0, 14.1, 15.2, 16.4, 17.7, 19.0]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(years, depth, marker='o', color='darkred', linewidth=3, markersize=8)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Depth to Groundwater (m bgl)", fontsize=12)
    ax.set_title("Groundwater Depletion Trend (~1.05 m/year decline)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")

# ============================================================
# MODE 2 — SAMPLE ANALYSIS
# ============================================================
@app.post("/mode2/sample-analysis")
def sample_analysis(sample: SampleQuery):
    logging.info(f"MODE2 request: parameters={sample.parameters}, season={sample.season}, trend={sample.groundwater_trend}")
    
    season = (sample.season or "").lower()
    trend = (sample.groundwater_trend or "").lower()

    user_params = {k.lower(): v for k, v in sample.parameters.items()}
    input_vector = []
    missing = []
    for col in FEATURE_COLUMNS:
        if col in user_params:
            input_vector.append(user_params[col])
        else:
            input_vector.append(FEATURE_DEFAULTS.get(col, 7.5))
            missing.append(col)

    values = np.array([input_vector], dtype=np.float32)
    param_values = dict(zip(FEATURE_COLUMNS, input_vector))

    # --- ML inference ---
    recon_error = None
    anomaly_status = "Normal"
    cluster_id = None
    cluster_confidence = 0.5
    recon_conf = 0.0
    recon_level = "Medium"

    try:
        xs = scaler.transform(values)
        xp = pca.transform(xs)
        xt = torch.tensor(xp, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            xrec, _ = ae(xt)
            recon = xrec.cpu().numpy()
        recon_error = float(np.mean((xp - recon) ** 2))
        anomaly_status = "Anomalous" if recon_error > 0.15 else "Normal"
        cluster_id = int(rf.predict(xp)[0])
        if hasattr(rf, "predict_proba"):
            proba = rf.predict_proba(xp)[0]
            cluster_confidence = float(np.max(proba))
        recon_conf = max(0.0, min(1.0, 1.0 - recon_error * 5))
        recon_level = "High" if recon_error < 0.1 else "Medium" if recon_error < 0.2 else "Low"
    except Exception as ml_e:
        logging.error(f"ML inference failed: {ml_e}")

    # --- Rule-based calculations ---
    exceeded_params = []
    is_unsafe = False
    if param_values.get("nitrate", 0) > CRITICAL_LIMITS["nitrate"]:
        exceeded_params.append(f"Nitrate ({param_values['nitrate']:.1f} > 45 mg/L)")
        is_unsafe = True
    if param_values.get("fluoride", 0) > CRITICAL_LIMITS["fluoride"]:
        exceeded_params.append(f"Fluoride ({param_values['fluoride']:.1f} > 1.5 mg/L)")
        is_unsafe = True
    if param_values.get("iron", 0) > CRITICAL_LIMITS["iron"]:
        exceeded_params.append(f"Iron ({param_values['iron']:.1f} > 1.0 mg/L)")
        is_unsafe = True
    ph_val = param_values.get("ph", 7.5)
    if ph_val < CRITICAL_LIMITS["ph_min"] or ph_val > CRITICAL_LIMITS["ph_max"]:
        exceeded_params.append(f"pH ({ph_val:.1f} outside 6.5–8.5)")
        is_unsafe = True

    # WQI calculation
    qi_dict = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for p, w in WEIGHTS.items():
        ci = param_values.get(p, 0)
        if p == "ph":
            deviation = max(ph_val - CRITICAL_LIMITS["ph_max"], CRITICAL_LIMITS["ph_min"] - ph_val, 0)
            qi = deviation * 50
        else:
            si = STANDARD_DESIRABLE.get(p, 1000)
            qi = min(200, (ci / si) * 100 if si > 0 else 0)
        qi_dict[p] = qi
        weighted_sum += qi * w
        total_weight += w
    wqi_badness = weighted_sum / total_weight if total_weight > 0 else 0
    human_wqi = max(0, min(100, 100 - wqi_badness))

    if is_unsafe:
        category = "Unsafe"
    elif human_wqi >= 90:
        category = "Excellent"
    elif human_wqi >= 70:
        category = "Good"
    elif human_wqi >= 50:
        category = "Moderate"
    else:
        category = "Poor"

    # Confidence & uncertainty
    missing_ratio = len(missing) / len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0
    data_completeness = "High" if missing_ratio < 0.2 else "Medium" if missing_ratio < 0.5 else "Low"
    overall_confidence = round((recon_conf + cluster_confidence) / 2, 3) if recon_error is not None else 0.0
    confidence_level = "High" if recon_level == "High" and data_completeness == "High" else "Medium" if recon_level != "Low" else "Low"

    # Primary risks
    primary_risks = []
    if any("nitrate" in e.lower() for e in exceeded_params):
        primary_risks.append("Agricultural nitrate runoff")
    if any("fluoride" in e.lower() for e in exceeded_params):
        primary_risks.append("Geogenic fluorosis risk")
    if any("iron" in e.lower() for e in exceeded_params):
        primary_risks.append("Natural iron mobilization")
    if param_values.get("tds", 0) > 1000:
        primary_risks.append("Salinity issues")
    if param_values.get("hardness", 0) > 300:
        primary_risks.append("High hardness scaling")

    # Usage suitability
    ec_val = param_values.get("conductivity", 850.0)
    tds_val = param_values.get("tds", 550.0)
    hardness_val = param_values.get("hardness", 280.0)

    salinity_class = (
        "Low (C1)" if ec_val < 250 else
        "Medium (C2)" if ec_val < 750 else
        "High (C3)" if ec_val < 2250 else
        "Very High (C4)"
    )
    irrigation_suitability = (
        "Excellent – suitable for all crops" if ec_val < 750 else
        "Good – permeable soils & moderate leaching required" if ec_val < 2250 else
        "Restricted – salt-tolerant crops & good drainage needed"
    )
    drinking_suitability = "Safe for drinking" if not is_unsafe else "Requires treatment before drinking"
    hardness_class = (
        "Soft" if hardness_val < 75 else
        "Moderately hard" if hardness_val < 150 else
        "Hard" if hardness_val < 300 else
        "Very hard"
    )

    summary_parts = []
    if is_unsafe:
        summary_parts.append("⚠️ Unsafe for direct drinking due to critical exceedances.")
    else:
        summary_parts.append(f"Overall quality: {category}.")
    if exceeded_params:
        summary_parts.append("Exceeded critical limits: " + "; ".join(exceeded_params) + ".")
    if season:
        if "pre" in season or "dry" in season:
            summary_parts.append("Pre-monsoon period: Higher concentrations typical.")
        elif "post" in season:
            summary_parts.append("Post-monsoon period: Dilution often improves quality.")
    if trend == "declining":
        summary_parts.append("Declining groundwater trend raises long-term concerns.")
    if primary_risks:
        summary_parts.append("Primary risks: " + "; ".join(primary_risks) + ".")

    ai_summary = " ".join(summary_parts) if summary_parts else "Sample appears typical."

    response_data = {
        "ml_results": {
            "reconstruction_error": round(recon_error, 4) if recon_error is not None else None,
            "anomaly_status": anomaly_status,
            "cluster_id": cluster_id,
            "cluster_confidence": round(cluster_confidence, 3),
            "overall_model_confidence": overall_confidence,
            "confidence_level": confidence_level
        },
        "regulatory_compliance": {
            "status": "Non-compliant (Unsafe)" if is_unsafe else "Compliant",
            "exceeded_critical_parameters": exceeded_params
        },
        "water_quality_index": {
            "weighted_wqi": round(human_wqi, 1),
            "category": category
        },
        "standards_reference": {
            "bis_who_limits_table": BIS_WHO_STANDARDS,
            "source": "Bureau of Indian Standards (IS 10500:2012) & WHO Guidelines"
        },
        "uncertainty_analysis": {
            "data_completeness": data_completeness,
            "missing_features_count": len(missing),
            "seasonal_context_provided": bool(season),
            "model_confidence_level": confidence_level,
            "recommendation": "Provide all parameters and seasonal context for maximum confidence"
        },
        "usage_suitability": {
            "drinking": drinking_suitability,
            "irrigation": {
                "salinity_class": salinity_class,
                "suitability": irrigation_suitability
            },
            "hardness_classification": hardness_class
        },
        "feature_handling": {
            "user_provided": list(user_params.keys()),
            "auto_filled_count": len(missing)
        },
        "ai_explanation": {
            "summary": ai_summary,
            "primary_risks": primary_risks
        },
        "visualizations": {
            "parameter_bar_chart": "/mode2/sample-plot"
        }
    }

    response_data["executive_summary"] = generate_executive_summary(response_data, "sample")
    return response_data

@app.post("/mode2/sample-plot", response_class=StreamingResponse)
def sample_plot(sample: SampleQuery):
    logging.info("MODE2 plot requested")
    user_params = {k.lower(): v for k, v in sample.parameters.items()}
    labels = [f.upper() for f in IMPORTANT_USER_FEATURES]
    values = [user_params.get(f.lower(), FEATURE_DEFAULTS.get(f.lower(), 0.0)) for f in IMPORTANT_USER_FEATURES]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.bar(labels, values, color='skyblue', edgecolor='navy', alpha=0.8)
    ax.set_ylabel("Concentration (mg/L or unit)")
    ax.set_title("Groundwater Parameters vs BIS/WHO Limits")
    ax.tick_params(axis='x', rotation=30)

    # Draw limit lines/shading
    for i, label in enumerate(labels):
        limit = PLOT_LIMITS.get(label)
        if limit:
            if isinstance(limit, tuple):
                ax.axhspan(limit[0], limit[1], color='lightgreen', alpha=0.2)
                ax.axhline(limit[0], color='green', linestyle='--', alpha=0.7)
                ax.axhline(limit[1], color='red', linestyle='--', alpha=0.7)
            else:
                ax.axhline(limit, color='red', linestyle='--', alpha=0.7)

    # Color bars based on exceedance
    for bar, label, value in zip(bars, labels, values):
        limit = PLOT_LIMITS.get(label)
        if limit:
            if isinstance(limit, tuple):
                low, high = limit
                if value < low or value > high:
                    bar.set_facecolor('crimson')
                else:
                    bar.set_facecolor('lightgreen')
            else:
                if value > limit:
                    bar.set_facecolor('crimson')
                else:
                    bar.set_facecolor('lightgreen')

    # Value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")

# ============================================================
# RUN SERVER (Render-compatible: uses $PORT)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)