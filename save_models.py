"""Save trained models and pre-computed results for the demo app."""
import json
import joblib
import numpy as np
import pandas as pd
import os
from ltc_models import (
    load_and_prepare_data,
    train_frequency_glm, train_frequency_xgb,
    train_frequency_lgbm, train_frequency_rf,
    train_severity_xgb, train_severity_lgbm, train_severity_rf,
    train_tweedie, train_tweedie_xgb, train_tweedie_lgbm,
    evaluate_frequency, evaluate_severity, evaluate_pure_premium,
    get_frequency_proba,
)

def main():
    os.makedirs("saved_models", exist_ok=True)

    print("Loading data...")
    df, X, features, idx_tr, idx_te = load_and_prepare_data()
    X_train, X_test = X.loc[idx_tr], X.loc[idx_te]
    yf_tr = df.loc[idx_tr, "Claim_Occurred"]
    yf_te = df.loc[idx_te, "Claim_Occurred"]

    # --- Train frequency models ---
    print("\n=== Frequency Models ===")
    freq_models = {}
    freq_params = {}
    for name, fn in [("GLM", train_frequency_glm), ("XGBoost", train_frequency_xgb),
                     ("LightGBM", train_frequency_lgbm), ("RandomForest", train_frequency_rf)]:
        print(f"  Training {name}...")
        model, params = fn(X_train, yf_tr)
        freq_models[name] = model
        freq_params[name] = params

    # Evaluate frequency
    freq_results = []
    for name, model in freq_models.items():
        p = get_frequency_proba(model, X_test)
        freq_results.append(evaluate_frequency(yf_te, p, model_name=name))
        print(f"    {name}: Gini={freq_results[-1]['Gini']:.4f}")

    # --- Train severity models ---
    print("\n=== Severity Models ===")
    clm_tr = idx_tr[df.loc[idx_tr, "Claim_Occurred"] == 1]
    clm_te = idx_te[df.loc[idx_te, "Claim_Occurred"] == 1]
    ys_tr = df.loc[clm_tr, "Total_LTC_Payout_USD"]
    ys_te = df.loc[clm_te, "Total_LTC_Payout_USD"]

    sev_models = {}
    sev_params = {}
    for name, fn in [("XGBoost_Gamma", train_severity_xgb),
                     ("LightGBM_Gamma", train_severity_lgbm),
                     ("RandomForest", train_severity_rf)]:
        print(f"  Training {name}...")
        model, params = fn(X.loc[clm_tr], ys_tr)
        sev_models[name] = model
        sev_params[name] = params

    # Evaluate severity
    sev_results = []
    for name, model in sev_models.items():
        pred = model.predict(X.loc[clm_te])
        sev_results.append(evaluate_severity(ys_te.values, pred, model_name=name))
        print(f"    {name}: MAE=${sev_results[-1]['MAE']:,.0f}")

    # --- Tweedie models ---
    print("\n=== Tweedie (Single-Model) ===")
    y_loss_tr = df.loc[idx_tr, "Total_LTC_Payout_USD"]
    y_loss_te = df.loc[idx_te, "Total_LTC_Payout_USD"]

    tw_glm, tw_glm_params = train_tweedie(X_train, y_loss_tr)
    tw_xgb, tw_xgb_params = train_tweedie_xgb(X_train, y_loss_tr)
    tw_lgbm, tw_lgbm_params = train_tweedie_lgbm(X_train, y_loss_tr)

    tweedie_results = []
    for name, model in [("Tweedie GLM", tw_glm), ("Tweedie XGBoost", tw_xgb), ("Tweedie LightGBM", tw_lgbm)]:
        pred = model.predict(X_test)
        r = evaluate_pure_premium(y_loss_te.values, pred, name)
        tweedie_results.append(r)
        print(f"    {name}: Gini={r['Normalized_Gini']:.4f} Calib={r['Calibration_Ratio']:.4f}")

    # --- Pure premium combinations ---
    print("\n=== Pure Premium Combinations ===")
    actual_loss = y_loss_te.values
    pp_results = []
    sev_name_map = {"XGBoost_Gamma": "XGBoost (Gamma)", "LightGBM_Gamma": "LightGBM (Gamma)",
                    "RandomForest": "Random Forest"}

    for f_name, f_model in freq_models.items():
        p_freq = get_frequency_proba(f_model, X_test)
        for s_name, s_model in sev_models.items():
            sev_all = s_model.predict(X_test)
            pp = p_freq * sev_all
            combo = f"{f_name} + {sev_name_map.get(s_name, s_name)}"
            pp_results.append(evaluate_pure_premium(actual_loss, pp, combo))

    # Add all Tweedie results
    pp_results.extend(tweedie_results)

    # --- Feature importances ---
    feat_imp = {}
    for name, model in freq_models.items():
        if hasattr(model, "feature_importances_"):
            feat_imp[f"freq_{name}"] = dict(zip(features, model.feature_importances_.tolist()))
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0])
            feat_imp[f"freq_{name}"] = dict(zip(features, (coefs / coefs.sum()).tolist()))
    for name, model in sev_models.items():
        if hasattr(model, "feature_importances_"):
            feat_imp[f"sev_{name}"] = dict(zip(features, model.feature_importances_.tolist()))

    # --- Save models ---
    print("\nSaving models...")
    joblib.dump(freq_models["GLM"], "saved_models/freq_glm.joblib")
    joblib.dump(freq_models["XGBoost"], "saved_models/freq_xgb.joblib")
    joblib.dump(sev_models["XGBoost_Gamma"], "saved_models/sev_xgb.joblib")
    # Save best Tweedie (GLM has best calibration + gini)
    joblib.dump(tw_glm, "saved_models/tweedie.joblib")

    # --- Save results JSON ---
    results = {
        "features": features,
        "frequency_comparison": freq_results,
        "severity_comparison": sev_results,
        "pure_premium_comparison": pp_results,
        "feature_importances": feat_imp,
        "best_freq_params": {k: v for k, v in freq_params.items() if v},
        "best_sev_params": {k: v for k, v in sev_params.items() if v},
        "tweedie_params": {"GLM": tw_glm_params, "XGBoost": tw_xgb_params, "LightGBM": tw_lgbm_params},
        "dataset_stats": {
            "total_policies": int(df.shape[0]),
            "claim_rate": float(df.Claim_Occurred.mean()),
            "mean_severity": float(df.loc[df.Claim_Occurred==1, "Total_LTC_Payout_USD"].mean()),
            "median_severity": float(df.loc[df.Claim_Occurred==1, "Total_LTC_Payout_USD"].median()),
        }
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open("saved_models/results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # --- Save sample data ---
    claims = df[df.Claim_Occurred == 1].sample(10, random_state=42)
    no_claims = df[df.Claim_Occurred == 0].sample(10, random_state=42)
    samples = pd.concat([claims, no_claims]).reset_index(drop=True)
    samples.to_json("saved_models/sample_data.json", orient="records", indent=2)

    print("\nAll saved to saved_models/")
    print("Done!")

if __name__ == "__main__":
    main()
