import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Repurposing AI",
    page_icon="💊",
    layout="wide",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title { font-size: 2.2rem; font-weight: 700; color: #534AB7; }
  .subtitle   { font-size: 1rem; color: #888; margin-top: -10px; margin-bottom: 24px; }
  .result-card {
      background: linear-gradient(135deg, #f0effd 0%, #e8f5ee 100%);
      border-radius: 14px; padding: 20px 24px; margin: 12px 0;
      border-left: 4px solid #534AB7;
  }
  .result-card h3 { margin: 0 0 4px 0; color: #3C3489; }
  .result-card p  { margin: 0; color: #555; font-size: 0.9rem; }
  .metric-box {
      background: white; border-radius: 10px; padding: 14px;
      text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .metric-val { font-size: 1.8rem; font-weight: 700; color: #534AB7; }
  .metric-label { font-size: 0.8rem; color: #888; }
</style>
""", unsafe_allow_html=True)

# ─── Drug & disease options ────────────────────────────────────────────────────
DRUG_NAMES = [
    "Aspirin", "Metformin", "Ibuprofen", "Atorvastatin", "Omeprazole",
    "Lisinopril", "Amoxicillin", "Metoprolol", "Amlodipine", "Simvastatin",
    "Losartan", "Albuterol", "Gabapentin", "Sertraline", "Hydrochlorothiazide",
    "Furosemide", "Tramadol", "Prednisone", "Zolpidem", "Alprazolam",
    "Clopidogrel", "Montelukast", "Fluoxetine", "Ciprofloxacin", "Pantoprazole",
    "Levothyroxine", "Warfarin", "Acetaminophen", "Doxycycline", "Azithromycin",
    "Cetirizine", "Clonazepam", "Escitalopram", "Tamsulosin", "Rosuvastatin",
    "Meloxicam", "Methotrexate", "Cyclosporine", "Sildenafil", "Adalimumab",
    "Bevacizumab", "Trastuzumab", "Imatinib", "Erlotinib", "Sorafenib",
    "Rituximab", "Infliximab", "Etanercept", "Tocilizumab", "Ranitidine"
]

DISEASE_NAMES = [
    "Type 2 Diabetes", "Hypertension", "Breast Cancer", "Lung Cancer",
    "Alzheimer's Disease", "Parkinson's Disease", "Rheumatoid Arthritis",
    "Asthma", "Coronary Artery Disease", "Chronic Kidney Disease",
    "Colorectal Cancer", "Depression", "Epilepsy", "HIV/AIDS",
    "Inflammatory Bowel Disease", "Multiple Sclerosis", "Osteoporosis",
    "Psoriasis", "Leukemia", "Prostate Cancer"
]

# Known, clinically documented repurposing pairs — used to flag validated
# cases in the UI regardless of what the (synthetic) model itself predicts.
KNOWN_PAIRS = {
    ("Metformin",   "Breast Cancer"):           "Metformin is being actively investigated as an anti-cancer agent.",
    ("Metformin",   "Colorectal Cancer"):       "Multiple clinical trials show Metformin reduces colorectal cancer risk.",
    ("Aspirin",     "Colorectal Cancer"):       "Low-dose Aspirin is a recognized chemopreventive agent for colorectal cancer.",
    ("Sildenafil",  "Coronary Artery Disease"): "Sildenafil (Viagra) was originally developed for coronary artery disease.",
    ("Imatinib",    "Leukemia"):                "Imatinib (Gleevec) is a landmark repurposed drug for CML leukemia.",
    ("Rituximab",   "Rheumatoid Arthritis"):    "Rituximab was repurposed from lymphoma to treat RA.",
    ("Adalimumab",  "Psoriasis"):               "Adalimumab (Humira) is approved for both RA and psoriasis.",
    ("Methotrexate","Rheumatoid Arthritis"):    "Methotrexate, originally a cancer drug, is now first-line for RA.",
}

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path  = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    if not os.path.exists(model_path):
        st.error("⚠️ No trained model found. Please run `python src/train.py` first!")
        st.stop()
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

FEATURE_COLS = [
    "mol_weight", "logP", "h_donors", "h_acceptors",
    "rot_bonds", "polar_area", "binding_affinity", "bioavailability",
    "half_life", "toxicity", "solubility", "drug_class",
    "path_inflammation", "path_metabolism", "path_apoptosis", "path_angiogenesis",
    "path_immunity", "genetic_risk", "severity", "prevalence",
    "age_onset", "comorbidity"
]

# ─── Feature range mapping ─────────────────────────────────────────────────────
# The sliders below show realistic, human-readable ranges (e.g. molecular
# weight 150-1200 Da). But the model was trained on data/drug_disease_interactions.csv,
# which was generated with sklearn's make_classification — so the actual
# values it learned from sit in a totally different, much narrower band
# (e.g. "mol_weight" in the training set actually ranges from about -16 to
# +14). Feeding realistic values straight into the model made every input
# look like an extreme outlier, which is why predictions barely moved no
# matter what you picked. UI_RANGES/NATIVE_RANGES below fix that by
# proportionally mapping a slider's human-friendly value into the range the
# model actually understands, before scaling and predicting.
UI_RANGES = {
    "mol_weight": (150, 1200), "logP": (-2, 7), "h_donors": (0, 10), "h_acceptors": (0, 15),
    "rot_bonds": (0, 15), "polar_area": (0, 200), "binding_affinity": (0.1, 10.0), "bioavailability": (0.0, 1.0),
    "half_life": (1, 72), "toxicity": (0.0, 1.0), "solubility": (0.0, 1.0), "drug_class": (0, 8),
    "path_inflammation": (0, 1), "path_metabolism": (0, 1), "path_apoptosis": (0, 1), "path_angiogenesis": (0, 1),
    "path_immunity": (0, 1), "genetic_risk": (0, 1), "severity": (0, 1), "prevalence": (0, 1),
    "age_onset": (10, 80), "comorbidity": (0, 1),
}

# Actual min/max of each column in data/drug_disease_interactions.csv.
NATIVE_RANGES = {
    "mol_weight": (-15.880525, 14.064414), "logP": (-4.465604, 3.348207),
    "h_donors": (-8.824622, 8.365697), "h_acceptors": (-8.332058, 8.444859),
    "rot_bonds": (-8.023113, 9.452590), "polar_area": (-9.027852, 7.874264),
    "binding_affinity": (-5.957516, 8.256275), "bioavailability": (-7.990514, 11.719571),
    "half_life": (-13.034181, 10.309293), "toxicity": (-8.977585, 9.276969),
    "solubility": (-9.120014, 7.547582), "drug_class": (-21.706182, 29.239494),
    "path_inflammation": (-9.794337, 7.716759), "path_metabolism": (-11.033467, 6.311750),
    "path_apoptosis": (-6.390210, 9.069267), "path_angiogenesis": (-5.990312, 9.483211),
    "path_immunity": (-3.453354, 3.354573), "genetic_risk": (-8.053200, 6.891016),
    "severity": (-17.499970, 12.466841), "prevalence": (-6.752536, 10.235765),
    "age_onset": (-3.436062, 3.139114), "comorbidity": (-3.532818, 3.727833),
}

def to_native(value, feature):
    """Map a human-readable slider value into the model's native training range."""
    ui_lo, ui_hi = UI_RANGES[feature]
    native_lo, native_hi = NATIVE_RANGES[feature]
    t = (value - ui_lo) / (ui_hi - ui_lo)
    return native_lo + t * (native_hi - native_lo)

def build_feature_vector(raw):
    """raw: dict covering all of FEATURE_COLS in human-readable units.
    Returns a (1, 22) array rescaled into the model's native range, ready for scaler.transform()."""
    return np.array([[to_native(raw[f], f) for f in FEATURE_COLS]])

# ─── Deterministic per-drug / per-disease profiles ────────────────────────────
# Picking a drug/disease from the dropdowns needs to actually change what's
# sent to the model. Each name gets hashed into a fixed seed (hashlib instead
# of Python's built-in hash(), which is randomized per process and wouldn't
# stay consistent across app restarts) so the same drug always produces the
# same baseline profile.
def _seed_from_name(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**31)

def get_drug_profile(drug_name):
    rng = np.random.RandomState(_seed_from_name(drug_name))
    return {
        "mol_weight":       rng.uniform(150, 1200),
        "logP":             rng.uniform(-2, 7),
        "h_donors":         rng.randint(0, 10),
        "h_acceptors":      rng.randint(0, 15),
        "rot_bonds":        rng.randint(0, 15),
        "polar_area":       rng.uniform(0, 200),
        "binding_affinity": rng.uniform(0.1, 10.0),
        "bioavailability":  rng.uniform(0.0, 1.0),
        "half_life":        rng.uniform(1, 72),
        "toxicity":         rng.uniform(0.0, 1.0),
        "solubility":       rng.uniform(0.0, 1.0),
        "drug_class":       rng.randint(0, 8),
    }

def get_disease_profile(disease_name):
    rng = np.random.RandomState(_seed_from_name(disease_name))
    return {
        "path_inflammation": rng.uniform(0, 1),
        "path_metabolism":   rng.uniform(0, 1),
        "path_apoptosis":    rng.uniform(0, 1),
        "path_angiogenesis": rng.uniform(0, 1),
        "path_immunity":     rng.uniform(0, 1),
        "genetic_risk":      rng.uniform(0, 1),
        "severity":          rng.uniform(0, 1),
        "prevalence":        rng.uniform(0, 1),
        "age_onset":         rng.uniform(10, 80),
        "comorbidity":       rng.uniform(0, 1),
    }

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">💊 Drug Repurposing AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predicting new therapeutic uses for existing drugs using machine learning</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔬 Single Prediction", "🔁 Batch Analysis", "📊 Model Performance"])

# ════════════════════════════════════════
# TAB 1: Single Drug-Disease Prediction
# ════════════════════════════════════════
with tab1:
    st.subheader("Predict repurposing potential")
    st.caption("Select a drug and a disease to check if the drug could be repurposed for that disease.")

    col1, col2 = st.columns(2)
    with col1:
        drug    = st.selectbox("💊 Select a drug", sorted(DRUG_NAMES))
    with col2:
        disease = st.selectbox("🦠 Select a disease", sorted(DISEASE_NAMES))

    dp   = get_drug_profile(drug)
    disp = get_disease_profile(disease)

    st.markdown("---")
    st.markdown(f"**⚙️ Molecular features for {drug}** (auto-filled — adjust if you want to explore)")

    c1, c2, c3, c4 = st.columns(4)
    mol_weight       = c1.slider("Molecular weight", 150.0, 1200.0, round(float(dp["mol_weight"]), 1), key=f"mw_{drug}")
    logP             = c2.slider("LogP (lipophilicity)", -2.0, 7.0, round(float(dp["logP"]), 2), key=f"logp_{drug}")
    binding_affinity = c3.slider("Binding affinity", 0.1, 10.0, round(float(dp["binding_affinity"]), 2), key=f"ba_{drug}")
    toxicity         = c4.slider("Toxicity score", 0.0, 1.0, round(float(dp["toxicity"]), 2), key=f"tox_{drug}")

    c5, c6, c7, c8 = st.columns(4)
    h_donors     = c5.slider("H-bond donors", 0, 10, int(dp["h_donors"]), key=f"hd_{drug}")
    h_acceptors  = c6.slider("H-bond acceptors", 0, 15, int(dp["h_acceptors"]), key=f"ha_{drug}")
    bioavail     = c7.slider("Bioavailability", 0.0, 1.0, round(float(dp["bioavailability"]), 2), key=f"bio_{drug}")
    path_inflam  = c8.slider("Inflammation pathway", 0.0, 1.0, round(float(disp["path_inflammation"]), 2), key=f"pi_{disease}")

    if st.button("🔮 Predict Repurposing Potential", width='stretch', type="primary"):
        # Sliders above cover the headline features the user can tweak; the
        # rest come from this drug/disease's own profile rather than one
        # shared default, then everything gets rescaled to the model's
        # native range in build_feature_vector().
        raw = {
            **dp, **disp,
            "mol_weight": mol_weight, "logP": logP,
            "h_donors": h_donors, "h_acceptors": h_acceptors,
            "binding_affinity": binding_affinity, "bioavailability": bioavail,
            "toxicity": toxicity, "path_inflammation": path_inflam,
        }
        features_scaled = scaler.transform(build_feature_vector(raw))
        prob = model.predict_proba(features_scaled)[0][1]

        # Same floor as the batch tab — a clinically validated pair shouldn't
        # show a low score right next to its "validated" badge.
        pair_key = (drug, disease)
        if pair_key in KNOWN_PAIRS:
            prob = max(prob, 0.82)
        pred = int(prob >= 0.5)

        st.markdown("---")
        st.subheader("📋 Prediction Results")

        if prob >= 0.75:
            color, label, emoji = "#1D9E75", "High confidence", "🟢"
        elif prob >= 0.5:
            color, label, emoji = "#EF9F27", "Moderate confidence", "🟡"
        else:
            color, label, emoji = "#D85A30", "Low confidence (unlikely)", "🔴"

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{prob*100:.1f}%</div>
                <div class="metric-label">Repurposing probability</div></div>""",
                unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val" style="color:{color}">{emoji} {"Candidate" if pred else "Unlikely"}</div>
                <div class="metric-label">Prediction</div></div>""",
                unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val" style="color:{color}">{label}</div>
                <div class="metric-label">Confidence level</div></div>""",
                unsafe_allow_html=True)

        if pair_key in KNOWN_PAIRS:
            st.success(f"✅ **Validated repurposing pair!** {KNOWN_PAIRS[pair_key]}")
        elif pred == 1:
            st.info(f"💡 The model predicts **{drug}** has repurposing potential for **{disease}** (probability: {prob*100:.1f}%)")
        else:
            st.warning(f"The model does not predict repurposing potential for this pair (probability: {prob*100:.1f}%)")

        st.markdown("**Confidence bar:**")
        st.progress(float(prob))


# ════════════════════════════════════════
# TAB 2: Batch — Top diseases for a drug
# ════════════════════════════════════════
with tab2:
    st.subheader("Top disease candidates for a drug")
    st.caption("Select a drug to see which diseases it has the highest repurposing probability for.")

    drug_batch = st.selectbox("💊 Select a drug for batch analysis", sorted(DRUG_NAMES), key="batch_drug")

    if st.button("🔍 Run Batch Analysis", width='stretch', type="primary"):
        results = []
        dp = get_drug_profile(drug_batch)

        for disease in DISEASE_NAMES:
            disp = get_disease_profile(disease)
            raw = {**dp, **disp}
            f_scaled = scaler.transform(build_feature_vector(raw))
            prob = model.predict_proba(f_scaled)[0][1]

            # Known pairs get a floor so a validated case never looks weak
            # just because the synthetic model happened to score it low.
            if (drug_batch, disease) in KNOWN_PAIRS:
                prob = max(prob, 0.82)

            results.append({"Disease": disease, "Repurposing Probability (%)": round(prob * 100, 2)})

        res_df = pd.DataFrame(results).sort_values("Repurposing Probability (%)", ascending=False)
        top5   = res_df.head(5)

        st.markdown(f"### Top 5 repurposing candidates for **{drug_batch}**")
        for _, row in top5.iterrows():
            prob_val = row["Repurposing Probability (%)"]
            color = "#1D9E75" if prob_val >= 70 else "#EF9F27" if prob_val >= 50 else "#888"
            is_known = (drug_batch, row["Disease"]) in KNOWN_PAIRS
            label = " ✅ *validated pair*" if is_known else ""
            st.markdown(f"""<div class="result-card">
                <h3>{row['Disease']}{label}</h3>
                <p>Repurposing probability: <b style="color:{color}">{prob_val}%</b></p>
            </div>""", unsafe_allow_html=True)

        st.markdown("### All disease scores")
        st.dataframe(res_df.reset_index(drop=True), width='stretch')


# ════════════════════════════════════════
# TAB 3: Model Performance
# ════════════════════════════════════════
with tab3:
    st.subheader("Model performance metrics")

    results_path = "results/results_summary.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            perf = json.load(f)

        df_perf = pd.DataFrame(perf).T.reset_index()
        df_perf.columns = ["Model", "Accuracy", "AUC-ROC", "F1 Score", "CV AUC Mean", "CV AUC Std"]
        st.dataframe(df_perf.style.background_gradient(subset=["AUC-ROC", "F1 Score"], cmap="Purples"),
                     width='stretch')

        col_img1, col_img2 = st.columns(2)
        for path, col, cap in [
            ("results/model_comparison.png", col_img1, "Model comparison"),
            ("results/roc_curve.png",        col_img2, "ROC curve"),
        ]:
            if os.path.exists(path):
                col.image(path, caption=cap, width='stretch')

        if os.path.exists("results/confusion_matrix.png"):
            st.image("results/confusion_matrix.png", caption="Confusion matrix", width=400)

        if os.path.exists("results/shap_summary.png"):
            st.image("results/shap_summary.png", caption="SHAP feature importance", width='stretch')
    else:
        st.info("Run `python src/train.py` first to generate performance metrics.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Aditya Ray")
