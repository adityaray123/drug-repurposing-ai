# 💊 AI-Based Drug Repurposing Prediction System

Predicting new therapeutic uses for existing FDA-approved drugs using machine learning.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)

🔗 **Live Demo:** https://drug-repurposing-ai.streamlit.app/

---

## Overview

Drug repurposing is the idea of finding new uses for drugs that are already approved and already sitting on pharmacy shelves. Since the safety data already exists, it's a far faster and cheaper path to a treatment than starting from scratch with a brand-new molecule.

This project turns that idea into a working ML pipeline. Given a drug and a disease, it estimates how likely that drug is to be a viable treatment for that disease, based on a set of molecular, pharmacokinetic, and biological pathway features. Four classifiers are trained and compared, the best one is picked using cross-validated AUC-ROC, and the whole thing is wrapped in a Streamlit app so it's actually usable: single-pair predictions, batch analysis across diseases, and a dashboard showing how each model performed.

## Features

- **Single Prediction** — pick a drug and a disease, tune the underlying molecular features, and get a repurposing probability with a confidence rating
- **Batch Analysis** — rank every candidate disease for a chosen drug by repurposing potential
- **Model Performance dashboard** — accuracy, AUC-ROC, F1, and cross-validation scores for every trained model, in one place
- **Known-pair recognition** — clinically documented repurposing cases (e.g. Metformin for breast cancer) get flagged automatically when predicted
- Custom-styled UI rather than the default Streamlit theme

## How It Works

```
Synthetic dataset generation
        ↓
22 engineered features (molecular, pharmacokinetic, pathway, disease-level)
        ↓
Train/test split + SMOTE for class balancing
        ↓
Train 4 models: Random Forest, SVM, Gradient Boosting, MLP
        ↓
5-fold cross-validation, best model picked by AUC-ROC
        ↓
Served through Streamlit (app/app.py)
```

The 22 features cover molecular weight, LogP, H-bond donors/acceptors, rotatable bonds, polar surface area, binding affinity, bioavailability, half-life, toxicity, solubility, drug class, five pathway-involvement scores (inflammation, metabolism, apoptosis, angiogenesis, immunity), genetic risk, disease severity, prevalence, age of onset, and comorbidity.

## Model Performance

| Model | Accuracy | AUC-ROC | F1 Score | CV AUC (mean ± std) |
|---|---|---|---|---|
| **SVM** 🏆 | 0.930 | **0.938** | 0.885 | 0.970 ± 0.013 |
| Neural Network (MLP) | 0.903 | 0.936 | 0.847 | 0.970 ± 0.013 |
| Gradient Boosting | 0.890 | 0.929 | 0.827 | 0.971 ± 0.011 |
| Random Forest | 0.898 | 0.924 | 0.829 | 0.970 ± 0.014 |

SVM with an RBF kernel came out on top and is what's shipped as `models/best_model.pkl`. Full metrics live in [`results/results_summary.json`](results/results_summary.json).

<p align="center">
  <img src="results/roc_curve.png" width="45%" alt="ROC Curve" />
  <img src="results/confusion_matrix.png" width="45%" alt="Confusion Matrix" />
</p>
<p align="center">
  <img src="results/model_comparison.png" width="70%" alt="Model Comparison" />
</p>

## Validated Repurposing Pairs

These are real, clinically documented repurposing cases baked into the dataset as ground truth. The app flags them with a ✅ whenever they show up in a prediction:

| Drug | Disease | Evidence |
|---|---|---|
| Metformin | Breast Cancer | Actively studied as an anti-cancer agent |
| Metformin | Colorectal Cancer | Multiple trials show reduced cancer risk |
| Aspirin | Colorectal Cancer | Recognized chemopreventive agent |
| Sildenafil | Coronary Artery Disease | Originally developed for CAD, before Viagra |
| Imatinib | Leukemia | Landmark repurposing case (Gleevec, CML) |
| Rituximab | Rheumatoid Arthritis | Repurposed from lymphoma treatment |
| Adalimumab | Psoriasis | Approved for both RA and psoriasis |
| Methotrexate | Rheumatoid Arthritis | Originally a cancer drug, now first-line for RA |

## Project Structure

```
drug-repurposing-ai/
├── app/
│   └── app.py                       ← Streamlit web app (inference + dashboard)
├── data/
│   └── drug_disease_interactions.csv
├── models/
│   ├── best_model.pkl                ← SVM, selected on AUC-ROC
│   ├── scaler.pkl
│   ├── SVM.pkl
│   ├── Random_Forest.pkl
│   ├── Gradient_Boosting.pkl
│   └── Neural_Network_MLP.pkl
├── results/
│   ├── results_summary.json
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── model_comparison.png
├── src/
│   ├── generate_data.py             ← synthetic dataset generator
│   └── train.py                     ← training + evaluation pipeline
├── requirements.txt                 ← app/runtime dependencies
├── requirements-train.txt           ← extra deps for retraining only
└── README.md
```

## Tech Stack

Python 3.9+, scikit-learn (Random Forest, SVM, Gradient Boosting, MLP), imbalanced-learn for SMOTE, SHAP for explainability, Streamlit for the app, plus pandas, numpy, and joblib doing the usual heavy lifting underneath.

## Setup & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/drug-repurposing-ai.git
cd drug-repurposing-ai

# 2. Install app dependencies
pip install -r requirements.txt

# 3. Launch the app (uses the pre-trained models already in models/)
streamlit run app/app.py
```

To regenerate the dataset or retrain the models from scratch:

```bash
pip install -r requirements.txt -r requirements-train.txt
python src/generate_data.py
python src/train.py
```

## Limitations & Disclaimer

The dataset here is synthetically generated, built to mimic realistic feature distributions and known repurposing biology, not pulled from real clinical trials or lab data. This is a proof-of-concept, not a validated pharmacological tool, and its predictions shouldn't be used for actual drug development, prescribing, or medical decisions.

## Future Improvements

- I may swap the synthetic data for a real bioactivity source, like DrugBank or ChEMBL.
- Will add option to add user's drugs option.
- Will add an option for to download the output.
- Improve the UI/Design.
- Add bigger Datasset.

## Author

Aditya Ray

---

<p align="center">Built with scikit-learn and Streamlit</p>
