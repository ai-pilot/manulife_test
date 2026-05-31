"""FastAPI backend for LTC Pure Premium Demo App."""
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add parent dir so we can import ltc_models if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent / "saved_models"

app = FastAPI(title="LTC Pure Premium API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load artifacts at startup ---
freq_glm = joblib.load(BASE / "freq_glm.joblib")
freq_xgb = joblib.load(BASE / "freq_xgb.joblib")
sev_xgb = joblib.load(BASE / "sev_xgb.joblib")
tweedie = joblib.load(BASE / "tweedie.joblib")

with open(BASE / "results.json") as f:
    RESULTS = json.load(f)
with open(BASE / "sample_data.json") as f:
    SAMPLES = json.load(f)

FEATURES = RESULTS["features"]


class PolicyInput(BaseModel):
    Customer_Age: float
    Max_Daily_Benefit_USD: float
    Risk_Score_Tier: float
    Caregiver_Availability_Index: float
    Macro_Inflation_Rate: float
    Prior_Claims_Count: float
    Care_Setting_Preference: str  # "Home Care", "Assisted Living", "Nursing Home"


def encode_input(data: PolicyInput):
    """Encode a single policy input into a DataFrame with proper feature names."""
    import pandas as pd
    age = data.Customer_Age
    row = {
        "Customer_Age": age,
        "Max_Daily_Benefit_USD": data.Max_Daily_Benefit_USD,
        "Risk_Score_Tier": data.Risk_Score_Tier,
        "Caregiver_Availability_Index": data.Caregiver_Availability_Index,
        "Macro_Inflation_Rate": data.Macro_Inflation_Rate,
        "Prior_Claims_Count": data.Prior_Claims_Count,
        "Age_x_Risk": age * data.Risk_Score_Tier,
        "High_Risk_Flag": 1.0 if data.Risk_Score_Tier >= 4 else 0.0,
        "Claims_Per_Year": data.Prior_Claims_Count / max(age - 39, 1),
        "Log_Benefit": float(np.log1p(data.Max_Daily_Benefit_USD)),
        "Low_Caregiver_x_Age": age if data.Caregiver_Availability_Index <= 2 else 0.0,
        "Setting_Home Care": 1.0 if data.Care_Setting_Preference == "Home Care" else 0.0,
        "Setting_Nursing Home": 1.0 if data.Care_Setting_Preference == "Nursing Home" else 0.0,
    }
    return pd.DataFrame([row])


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/samples")
def get_samples():
    return SAMPLES


@app.get("/api/model-comparison")
def get_model_comparison():
    return {
        "frequency": RESULTS["frequency_comparison"],
        "severity": RESULTS["severity_comparison"],
        "pure_premium": RESULTS["pure_premium_comparison"],
        "feature_importances": RESULTS["feature_importances"],
        "dataset_stats": RESULTS["dataset_stats"],
        "best_freq_params": RESULTS["best_freq_params"],
        "best_sev_params": RESULTS["best_sev_params"],
    }


@app.post("/api/predict")
def predict(policy: PolicyInput):
    X = encode_input(policy)

    # GLM frequency
    p_glm = float(freq_glm.predict_proba(X)[0, 1])
    # XGBoost frequency
    p_xgb = float(freq_xgb.predict_proba(X)[0, 1])
    # XGBoost severity (predicted for all, used with freq)
    sev_pred = float(sev_xgb.predict(X)[0])
    # Tweedie direct
    tw_pred = float(tweedie.predict(X)[0])

    # Pure premiums
    pp_glm = p_glm * sev_pred
    pp_xgb = p_xgb * sev_pred

    # Feature importance (GLM coefficients as contribution)
    glm_coefs = np.abs(freq_glm.coef_[0])
    glm_importance = (glm_coefs / glm_coefs.sum()).tolist()

    # XGBoost feature importance
    xgb_importance = freq_xgb.feature_importances_.tolist()

    # Per-feature contribution for this input (coefficient * value)
    raw_contrib = (freq_glm.coef_[0] * X.values[0]).tolist()

    return {
        "input": policy.model_dump(),
        "frequency": {
            "glm": round(p_glm, 4),
            "xgboost": round(p_xgb, 4),
        },
        "severity": round(sev_pred, 2),
        "pure_premium": {
            "glm_xgb": round(pp_glm, 2),
            "xgb_xgb": round(pp_xgb, 2),
            "tweedie": round(tw_pred, 2),
        },
        "feature_importance": {
            "features": FEATURES,
            "glm_global": glm_importance,
            "xgb_global": xgb_importance,
            "glm_contribution": raw_contrib,
        },
    }


@app.post("/api/report")
def generate_report(policy: PolicyInput):
    """Generate a portfolio-level Solvency & Capital Allocation Report (PDF)."""
    import datetime
    from fpdf import FPDF

    stats = RESULTS["dataset_stats"]
    n = stats["total_policies"]
    claim_rate = stats["claim_rate"]
    mean_sev = stats["mean_severity"]
    aggregate_el = round(n * claim_rate * mean_sev)
    inflation = policy.Macro_Inflation_Rate
    safety_pct = round(inflation * 300, 1)  # rough safety loading
    min_reserve = round(aggregate_el * (1 + safety_pct / 100))
    capital_buffer = min_reserve - aggregate_el

    # Per-policy example
    X = encode_input(policy)
    p_glm = float(freq_glm.predict_proba(X)[0, 1])
    sev_pred = float(sev_xgb.predict(X)[0])
    pp = p_glm * sev_pred

    def section_header(pdf, num, title):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(31, 111, 139)
        pdf.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ---- Cover ----
    pdf.add_page()
    pdf.set_fill_color(31, 111, 139)
    pdf.rect(0, 0, 210, 42, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_y(10)
    pdf.cell(0, 10, "Q2 Solvency & Capital Allocation Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "LTC Portfolio Expected Loss Analysis", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(0, 0, 0)
    pdf.set_y(52)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Report Date: {datetime.date.today()}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Classification: CONFIDENTIAL", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ---- 1. Portfolio Risk Exposure ----
    section_header(pdf, 1, "Portfolio Risk Exposure")
    pdf.multi_cell(0, 5, (
        f"Our valuation models have analyzed the active book of {n:,} LTC policies. "
        f"The aggregate Expected Loss (Pure Premium) for the upcoming fiscal year is "
        f"${aggregate_el / 1e6:,.1f} Million.\n\n"
        f"This figure is derived from the two-stage frequency-severity model:\n"
        f"  - Portfolio claim frequency: {claim_rate:.1%}\n"
        f"  - Mean claim severity (among claimants): ${mean_sev:,.0f}\n"
        f"  - Aggregate: {n:,} x {claim_rate:.1%} x ${mean_sev:,.0f} = ${aggregate_el / 1e6:,.1f}M"
    ))
    pdf.ln(5)

    # ---- 2. Driver Analysis ----
    section_header(pdf, 2, "Driver Analysis")
    pdf.multi_cell(0, 5, (
        "Primary risk drivers pushing Expected Loss above baseline:\n\n"
        "  - Age concentration: Claim frequency rises steeply with age. Policyholders aged "
        "70-85 carry claim rates 3-9x higher than those under 50, and this cohort represents "
        "a growing share of the active book.\n\n"
        f"  - Caregiver shortages: Regions with low Caregiver Availability Index (1-2) show "
        "elevated severity, as claimants are routed into higher-cost nursing home settings "
        "when home care aides are unavailable.\n\n"
        "  - Risk Tier 4 & 5 concentration: These tiers carry 2-3x the claim frequency of "
        "lower tiers and account for a disproportionate share of total expected loss."
    ))
    pdf.ln(5)

    # ---- 3. Capital Reserve Recommendation ----
    section_header(pdf, 3, "Capital Reserve Recommendation")
    pdf.multi_cell(0, 5, (
        f"Current macro inflation rate: {inflation:.1%}. "
        f"To maintain NAIC capital solvency compliance and absorb tail risk, "
        f"we recommend a {safety_pct:.0f}% safety loading on the aggregate Expected Loss.\n\n"
    ))

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(255, 248, 230)
    pdf.cell(90, 7, "Metric", border=1, fill=True)
    pdf.cell(90, 7, "Value", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for label, val in [
        ("Aggregate Expected Loss", f"${aggregate_el / 1e6:,.1f} Million"),
        (f"Safety Loading ({safety_pct:.0f}%)", f"${capital_buffer / 1e6:,.1f} Million"),
        ("Minimum Required Capital Reserve", f"${min_reserve / 1e6:,.1f} Million"),
    ]:
        pdf.cell(90, 7, label, border=1)
        pdf.cell(90, 7, val, border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(180, 50, 20)
    pdf.multi_cell(0, 5, (
        f"Action Required: Adjust underwriting base rates for Risk Tiers 4 & 5 to cover "
        f"the projected ${capital_buffer / 1e6:,.1f}M capital buffer."
    ))
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # ---- 4. Per-Policy Example ----
    section_header(pdf, 4, "Per-Policy Example")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, (
        f"Example: Age {policy.Customer_Age:.0f}, Risk Tier {policy.Risk_Score_Tier:.0f}, "
        f"{policy.Care_Setting_Preference}, Caregiver Index {policy.Caregiver_Availability_Index:.1f}\n\n"
        f"  Stage 1 (Frequency): {p_glm:.0%} probability of filing a claim\n"
        f"  Stage 2 (Severity):  ${sev_pred:,.0f} expected cost if claim occurs\n"
        f"  Expected Loss:       {p_glm:.2f} x ${sev_pred:,.0f} = ${pp:,.0f}\n\n"
        f"Business translation: the pure premium for this policyholder is ${pp:,.0f}. "
        f"Charging less than this amount means the company is mathematically expected to "
        f"lose money on this policy."
    ))
    pdf.ln(5)

    # ---- 5. Model Validation Summary ----
    section_header(pdf, 5, "Model Validation Summary")
    pdf.multi_cell(0, 5, (
        "Models used:\n"
        "  - Frequency: Logistic GLM (Gini ~0.54) - interpretable, auditable for filings\n"
        "  - Severity: XGBoost with Gamma objective - handles the heavy right tail\n"
        "  - Calibration ratio: ~0.95 (predicted vs actual aggregate loss)\n"
        "  - Validated against Tweedie GLM single-model benchmark\n\n"
        "Monitoring: quarterly recalibration, PSI-based drift detection, "
        "GLM retained as regulatory benchmark alongside GBM challenger."
    ))

    # ---- Footer ----
    pdf.ln(12)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, "Prepared by LTC Analytics Team", align="C")

    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=LTC_Solvency_Report.pdf"},
    )


# Serve React static build if it exists (for single-app deployment)
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if STATIC_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
