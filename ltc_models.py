"""
ltc_models.py — Reusable model functions for LTC Expected Loss (Pure Premium) modeling.

Contains:
  - Data loading & preparation
  - Frequency models (GLM, XGBoost, LightGBM, CatBoost, Random Forest) with grid search
  - Severity models  (XGBoost, LightGBM, CatBoost, Random Forest) with grid search
  - Single Tweedie model (direct pure-premium estimation)
  - Evaluation helpers (Gini, RMSE, MAE, calibration, lift chart)
  - Model comparison utilities
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, mean_squared_error, mean_absolute_error, make_scorer
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# ---------------------------------------------------------------------------

def load_and_prepare_data(csv_path="ltc_actuarial_take_home_dataset.csv",
                          test_size=0.25):
    """Load the LTC dataset, one-hot encode categoricals, and split."""
    df = pd.read_csv(csv_path)

    # One-hot encode care-setting preference
    df_enc = df.join(
        pd.get_dummies(df.Care_Setting_Preference, prefix="Setting", drop_first=True)
    )
    features = [
        "Customer_Age", "Max_Daily_Benefit_USD", "Risk_Score_Tier",
        "Caregiver_Availability_Index", "Macro_Inflation_Rate",
        "Prior_Claims_Count",
    ] + [c for c in df_enc.columns if c.startswith("Setting_")]

    X = df_enc[features].astype(float)

    idx_tr, idx_te = train_test_split(
        df.index, test_size=test_size, random_state=RANDOM_STATE,
        stratify=df.Claim_Occurred,
    )
    return df, X, features, idx_tr, idx_te


# ---------------------------------------------------------------------------
# 2. FREQUENCY MODELS  — P(Claim)
# ---------------------------------------------------------------------------

def train_frequency_glm(X_train, y_train):
    """Logistic Regression (GLM baseline) — no grid search needed."""
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, {}


def _grid_search(estimator, param_grid, X_train, y_train, scoring, cv=3):
    """Run GridSearchCV and return the best estimator + best params."""
    gs = GridSearchCV(
        estimator, param_grid, scoring=scoring,
        cv=cv, n_jobs=-1, refit=True,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def train_frequency_xgb(X_train, y_train):
    """XGBoost classifier with hyperparameter grid search."""
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
    }
    base = xgb.XGBClassifier(
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=RANDOM_STATE,
        verbosity=0,
    )
    model, best = _grid_search(base, param_grid, X_train, y_train, scoring="roc_auc")
    return model, best


def train_frequency_lgbm(X_train, y_train):
    """LightGBM classifier with hyperparameter grid search."""
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31],
    }
    base = lgb.LGBMClassifier(
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=-1,
    )
    model, best = _grid_search(base, param_grid, X_train, y_train, scoring="roc_auc")
    return model, best


def train_frequency_catboost(X_train, y_train):
    """CatBoost classifier with hyperparameter grid search."""
    param_grid = {
        "iterations": [200, 300],
        "depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    base = CatBoostClassifier(
        random_state=RANDOM_STATE, verbose=0,
    )
    model, best = _grid_search(base, param_grid, X_train, y_train, scoring="roc_auc")
    return model, best


def train_frequency_rf(X_train, y_train):
    """Random Forest classifier with hyperparameter grid search."""
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [5, 10],
    }
    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    model, best = _grid_search(base, param_grid, X_train, y_train, scoring="roc_auc")
    return model, best


# ---------------------------------------------------------------------------
# 3. SEVERITY MODELS  — E[Payout | Claim]
# ---------------------------------------------------------------------------

def train_severity_xgb(X_train, y_train):
    """XGBoost regressor (Gamma objective) with grid search."""
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
    }
    base = xgb.XGBRegressor(
        objective="reg:gamma", subsample=0.8,
        random_state=RANDOM_STATE, verbosity=0,
    )
    model, best = _grid_search(
        base, param_grid, X_train, y_train,
        scoring="neg_mean_absolute_error",
    )
    return model, best


def train_severity_lgbm(X_train, y_train):
    """LightGBM regressor (Gamma deviance) with grid search."""
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31],
    }
    base = lgb.LGBMRegressor(
        objective="gamma", subsample=0.8,
        random_state=RANDOM_STATE, verbosity=-1,
    )
    model, best = _grid_search(
        base, param_grid, X_train, y_train,
        scoring="neg_mean_absolute_error",
    )
    return model, best


def train_severity_catboost(X_train, y_train):
    """CatBoost regressor with grid search."""
    param_grid = {
        "iterations": [200, 300],
        "depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    base = CatBoostRegressor(
        loss_function="RMSE", random_state=RANDOM_STATE, verbose=0,
    )
    model, best = _grid_search(
        base, param_grid, X_train, y_train,
        scoring="neg_mean_absolute_error",
    )
    return model, best


def train_severity_rf(X_train, y_train):
    """Random Forest regressor with grid search."""
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [5, 10],
    }
    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    model, best = _grid_search(
        base, param_grid, X_train, y_train,
        scoring="neg_mean_absolute_error",
    )
    return model, best


# ---------------------------------------------------------------------------
# 4. SINGLE TWEEDIE MODEL — Direct Pure Premium
# ---------------------------------------------------------------------------

def train_tweedie(X_train, y_train):
    """
    Tweedie GLM that directly models the pure premium (zero-inflated continuous).
    Power parameter between 1 and 2 handles the mix of exact zeros and
    positive continuous payouts.
    Grid search over power and regularization alpha.
    """
    param_grid = {
        "power": [1.5, 1.6, 1.7, 1.8, 1.9],
        "alpha": [0.1, 1.0, 10.0],
    }
    base = TweedieRegressor(max_iter=5000)
    model, best = _grid_search(
        base, param_grid, X_train, y_train,
        scoring="neg_mean_absolute_error",
    )
    return model, best


def train_tweedie_xgb(X_train, y_train):
    """
    XGBoost with Tweedie objective (reg:tweedie). Handles zero-inflated
    continuous targets directly using gradient boosted trees — combines
    the power of tree-based models with the Tweedie distribution.
    Grid search over tweedie_variance_power, max_depth, learning_rate.
    """
    param_grid = {
        "tweedie_variance_power": [1.5, 1.6, 1.7, 1.8, 1.9],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.03, 0.05],
        "n_estimators": [200, 300],
    }
    base = xgb.XGBRegressor(
        objective="reg:tweedie",
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model, best = _grid_search(
        base, param_grid, X_train, y_train,
        scoring="neg_mean_absolute_error",
    )
    return model, best


def train_tweedie_lgbm(X_train, y_train):
    """
    LightGBM with Tweedie objective. Fast histogram-based boosting
    for direct pure premium estimation.
    """
    param_grid = {
        "tweedie_variance_power": [1.5, 1.6, 1.7, 1.8],
        "num_leaves": [31, 50],
        "learning_rate": [0.03, 0.05],
        "n_estimators": [200, 300],
    }
    base = lgb.LGBMRegressor(
        objective="tweedie",
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model, best = _grid_search(
        base, param_grid, X_train, y_train,
        scoring="neg_mean_absolute_error",
    )
    return model, best


# ---------------------------------------------------------------------------
# 5. EVALUATION HELPERS
# ---------------------------------------------------------------------------

def evaluate_frequency(y_true, y_prob, model_name="Model"):
    """Return dict with AUC, Gini, and predicted vs actual frequency."""
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    return {
        "Model": model_name,
        "AUC": round(auc, 4),
        "Gini": round(gini, 4),
        "Pred_Freq": round(y_prob.mean(), 4),
        "Actual_Freq": round(y_true.mean(), 4),
    }


def evaluate_severity(y_true, y_pred, model_name="Model"):
    """Return dict with RMSE and MAE."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "Model": model_name,
        "RMSE": round(rmse, 0),
        "MAE": round(mae, 0),
        "Mean_Actual": round(y_true.mean(), 0),
    }


def normalized_gini(actual, pred):
    """Normalized Gini coefficient for pure-premium rank-ordering."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    order = np.argsort(pred)[::-1]
    a = actual[order]
    lorenz = np.cumsum(a) / a.sum()
    g = lorenz.sum() / len(a) - 0.5

    order2 = np.argsort(actual)[::-1]
    a2 = actual[order2]
    lorenz2 = np.cumsum(a2) / a2.sum()
    gp = lorenz2.sum() / len(a2) - 0.5

    return round(g / gp, 4) if gp != 0 else 0.0


def evaluate_pure_premium(actual_loss, pred_premium, model_name="Model"):
    """Calibration ratio and normalized Gini for pure premium."""
    cal = pred_premium.sum() / actual_loss.sum()
    gini = normalized_gini(actual_loss, pred_premium)
    return {
        "Model": model_name,
        "Total_Predicted": round(pred_premium.sum(), 0),
        "Total_Actual": round(actual_loss.sum(), 0),
        "Calibration_Ratio": round(cal, 4),
        "Normalized_Gini": gini,
    }


# ---------------------------------------------------------------------------
# 6. COMPARISON & VISUALIZATION
# ---------------------------------------------------------------------------

def compare_models(results_list):
    """Build a comparison DataFrame from a list of evaluation dicts."""
    return pd.DataFrame(results_list).set_index("Model")


def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Bar chart of feature importances (works for tree models)."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
    else:
        return  # GLM doesn't have feature_importances_
    imp = imp.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.plot.barh(ax=ax, color="#1f6f8b")
    ax.invert_yaxis()
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_lift_chart(actual_loss, pred_premium, title="Lift Chart"):
    """Decile lift chart comparing predicted vs actual mean loss."""
    ev = pd.DataFrame({"pred": pred_premium, "actual": actual_loss})
    ev["decile"] = pd.qcut(
        ev.pred.rank(method="first"), 10, labels=False
    ) + 1
    by_dec = ev.groupby("decile").agg(
        mean_pred=("pred", "mean"), mean_actual=("actual", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_dec.decile - 0.18, by_dec.mean_actual, width=0.36,
           label="Actual loss", color="#1f6f8b")
    ax.bar(by_dec.decile + 0.18, by_dec.mean_pred, width=0.36,
           label="Predicted pure premium", color="#9bbcd1")
    ax.set_xlabel("Predicted-risk decile (10 = riskiest)")
    ax.set_ylabel("Mean loss per policy (USD)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

    lift = by_dec.mean_actual.iloc[-1] / max(by_dec.mean_actual.iloc[0], 1)
    print(f"Top vs bottom decile actual-loss lift: {lift:,.0f}x")


def plot_frequency_comparison(freq_results):
    """Side-by-side bar chart of Gini scores across frequency models."""
    df = pd.DataFrame(freq_results).set_index("Model")
    fig, ax = plt.subplots(figsize=(8, 4))
    df["Gini"].plot.bar(ax=ax, color="#1f6f8b", edgecolor="white")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Frequency Model Comparison — Gini")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    for i, v in enumerate(df["Gini"]):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_severity_comparison(sev_results):
    """Side-by-side bar chart of MAE across severity models."""
    df = pd.DataFrame(sev_results).set_index("Model")
    fig, ax = plt.subplots(figsize=(8, 4))
    df["MAE"].plot.bar(ax=ax, color="#1f6f8b", edgecolor="white")
    ax.set_ylabel("Mean Absolute Error (USD)")
    ax.set_title("Severity Model Comparison — MAE")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    for i, v in enumerate(df["MAE"]):
        ax.text(i, v + 1000, f"${v:,.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.show()


def get_frequency_proba(model, X):
    """Get predicted probabilities from a frequency model."""
    return model.predict_proba(X)[:, 1]
