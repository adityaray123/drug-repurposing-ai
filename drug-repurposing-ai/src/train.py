import pandas as pd
import numpy as np
import os, joblib, json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DATA_PATH   = "data/drug_disease_interactions.csv"
MODEL_DIR   = "models"
RESULTS_DIR = "results"

FEATURE_COLS = [
    "mol_weight", "logP", "h_donors", "h_acceptors",
    "rot_bonds", "polar_area", "binding_affinity", "bioavailability",
    "half_life", "toxicity", "solubility", "drug_class",
    "path_inflammation", "path_metabolism", "path_apoptosis", "path_angiogenesis",
    "path_immunity", "genetic_risk", "severity", "prevalence",
    "age_onset", "comorbidity"
]
TARGET_COL  = "interaction"

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# 1. Load & split data
# ─────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURE_COLS].values
    y  = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    print(f"   After SMOTE: {len(X_train_bal)} balanced training samples")
    return X_train_bal, X_test, y_train_bal, y_test, df


# ─────────────────────────────────────────
# 2. Define models
# ─────────────────────────────────────────
def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1,
            max_depth=5, random_state=42
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", solver="adam",
            max_iter=300, random_state=42, early_stopping=True
        ),
    }


# ─────────────────────────────────────────
# 3. Train & evaluate all models
# ─────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models  = get_models()
    results = {}

    print("\n" + "="*60)
    print("  TRAINING & EVALUATION")
    print("="*60)

    best_auc   = 0
    best_model = None
    best_name  = ""

    for name, model in models.items():
        print(f"\n⏳ Training {name}...")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1  = f1_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")

        results[name] = {
            "accuracy":    round(acc, 4),
            "auc_roc":     round(auc, 4),
            "f1_score":    round(f1, 4),
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "cv_auc_std":  round(cv_scores.std(), 4),
        }

        print(f"   Accuracy : {acc:.4f}")
        print(f"   AUC-ROC  : {auc:.4f}")
        print(f"   F1 Score : {f1:.4f}")
        print(f"   CV AUC   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Save model
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, f"{MODEL_DIR}/{safe_name}.pkl")

        if auc > best_auc:
            best_auc   = auc
            best_model = model
            best_name  = name

    print(f"\n🏆 Best model: {best_name} (AUC = {best_auc:.4f})")
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

    return results, best_model, best_name


# ─────────────────────────────────────────
# 4. Visualizations
# ─────────────────────────────────────────
def plot_results(results, best_model, X_test, y_test):
    from sklearn.metrics import roc_curve

    # ── Comparison bar chart ───────────────────────────────────────────────
    names   = list(results.keys())
    metrics = ["accuracy", "auc_roc", "f1_score"]
    colors  = ["#534AB7", "#1D9E75", "#EF9F27"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison — Drug Repurposing AI", fontsize=14, fontweight="bold")

    for ax, metric, color in zip(axes, metrics, colors):
        vals = [results[n][metric] for n in names]
        bars = ax.barh(names, vals, color=color, alpha=0.85)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── ROC curve for best model ───────────────────────────────────────────
    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#534AB7", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1],"--", color="gray", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Best Model")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Confusion matrix ───────────────────────────────────────────────────
    y_pred = best_model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["No Interaction","Interaction"],
                yticklabels=["No Interaction","Interaction"], ax=ax)
    ax.set_title("Confusion Matrix — Best Model")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n📊 Plots saved to {RESULTS_DIR}/")


# ─────────────────────────────────────────
# 5. Feature importance (SHAP)
# ─────────────────────────────────────────
def explain_model(best_model, X_test, model_name):
    try:
        import shap
        print("\n⏳ Computing SHAP values...")

        if "Random Forest" in model_name or "Gradient" in model_name:
            explainer = shap.TreeExplainer(best_model)
            shap_vals = explainer.shap_values(X_test[:200])
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            explainer = shap.KernelExplainer(
                best_model.predict_proba, X_test[:50]
            )
            shap_vals = explainer.shap_values(X_test[:50])
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

        plt.figure()
        shap.summary_plot(shap_vals, X_test[:200],
                          feature_names=FEATURE_COLS, show=False)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   SHAP summary saved to {RESULTS_DIR}/shap_summary.png")

    except Exception as e:
        print(f"   SHAP skipped: {e}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = load_data()

    results, best_model, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    plot_results(results, best_model, X_test, y_test)
    explain_model(best_model, X_test, best_name)

    # Save results summary
    with open(f"{RESULTS_DIR}/results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ All done! Check the models/ and results/ folders.")
    print("\n📋 Final results summary:")
    for model, metrics in results.items():
        print(f"   {model:30s} → AUC: {metrics['auc_roc']} | F1: {metrics['f1_score']}")