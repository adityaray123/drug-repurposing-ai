import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

np.random.seed(42)

# ─────────────────────────────────────────────
# Drug names (50 real FDA-approved drugs)
# ─────────────────────────────────────────────
DRUG_NAMES = [
    "Aspirin", "Metformin", "Ibuprofen", "Atorvastatin", "Omeprazole",
    "Lisinopril", "Amoxicillin", "Metoprolol", "Amlodipine", "Simvastatin",
    "Losartan", "Albuterol", "Gabapentin", "Sertraline", "Hydrochlorothiazide",
    "Furosemide", "Tramadol", "Prednisone", "Zolpidem", "Alprazolam",
    "Clopidogrel", "Montelukast", "Fluoxetine", "Ciprofloxacin", "Pantoprazole",
    "Levothyroxine", "Warfarin", "Acetaminophen", "Doxycycline", "Azithromycin",
    "Cetirizine", "Clonazepam", "Escitalopram", "Tamsulosin", "Rosuvastatin",
    "Meloxicam", "Ranitidine", "Methotrexate", "Cyclosporine", "Sildenafil",
    "Adalimumab", "Bevacizumab", "Trastuzumab", "Imatinib", "Erlotinib",
    "Sorafenib", "Rituximab", "Infliximab", "Etanercept", "Tocilizumab"
]

# ─────────────────────────────────────────────
# Disease names (20 diseases)
# ─────────────────────────────────────────────
DISEASE_NAMES = [
    "Type 2 Diabetes", "Hypertension", "Breast Cancer", "Lung Cancer",
    "Alzheimer's Disease", "Parkinson's Disease", "Rheumatoid Arthritis",
    "Asthma", "Coronary Artery Disease", "Chronic Kidney Disease",
    "Colorectal Cancer", "Depression", "Epilepsy", "HIV/AIDS",
    "Inflammatory Bowel Disease", "Multiple Sclerosis", "Osteoporosis",
    "Psoriasis", "Leukemia", "Prostate Cancer"
]

def generate_drug_features(n_drugs):
    """
    Generate synthetic drug features:
    - Molecular weight, LogP, H-bond donors/acceptors (Lipinski features)
    - Binding affinity score
    - Toxicity score
    - Drug class encoding
    """
    features = {
        "molecular_weight":    np.random.uniform(150, 1200, n_drugs),
        "logP":                np.random.uniform(-2, 7, n_drugs),
        "h_bond_donors":       np.random.randint(0, 10, n_drugs),
        "h_bond_acceptors":    np.random.randint(0, 15, n_drugs),
        "rotatable_bonds":     np.random.randint(0, 15, n_drugs),
        "polar_surface_area":  np.random.uniform(0, 200, n_drugs),
        "binding_affinity":    np.random.uniform(0.1, 10.0, n_drugs),
        "bioavailability":     np.random.uniform(0.0, 1.0, n_drugs),
        "half_life_hours":     np.random.uniform(1, 72, n_drugs),
        "toxicity_score":      np.random.uniform(0.0, 1.0, n_drugs),
        "solubility":          np.random.uniform(0.0, 1.0, n_drugs),
        "drug_class":          np.random.randint(0, 8, n_drugs),   # 8 drug categories
    }
    return pd.DataFrame(features)

def generate_disease_features(n_diseases):
    """
    Generate synthetic disease features:
    - Pathway involvement scores
    - Genetic risk factors
    - Severity and prevalence
    """
    features = {
        "pathway_inflammation":  np.random.uniform(0, 1, n_diseases),
        "pathway_metabolism":    np.random.uniform(0, 1, n_diseases),
        "pathway_apoptosis":     np.random.uniform(0, 1, n_diseases),
        "pathway_angiogenesis":  np.random.uniform(0, 1, n_diseases),
        "pathway_immunity":      np.random.uniform(0, 1, n_diseases),
        "genetic_risk_score":    np.random.uniform(0, 1, n_diseases),
        "severity_score":        np.random.uniform(0, 1, n_diseases),
        "prevalence_rate":       np.random.uniform(0, 1, n_diseases),
        "age_of_onset":          np.random.uniform(10, 80, n_diseases),
        "comorbidity_index":     np.random.uniform(0, 1, n_diseases),
    }
    return pd.DataFrame(features)

def generate_interaction_dataset(n_samples=2000):
    """
    Generate drug-disease pairs with interaction labels.
    Label = 1 means the drug can treat/repurpose for that disease.
    Uses make_classification for realistic feature overlap.
    """
    n_drugs    = len(DRUG_NAMES)
    n_diseases = len(DISEASE_NAMES)

    # Generate base features using sklearn (creates realistic class separability)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=22,            # 12 drug + 10 disease features
        n_informative=14,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=[0.7, 0.3],       # ~30% positive interactions (realistic)
        flip_y=0.05,              # 5% label noise for realism
        random_state=42
    )

    drug_features    = generate_drug_features(n_drugs)
    disease_features = generate_disease_features(n_diseases)

    # Assign random drug-disease pairs to each sample
    drug_indices    = np.random.randint(0, n_drugs, n_samples)
    disease_indices = np.random.randint(0, n_diseases, n_samples)

    df = pd.DataFrame(X, columns=[
        "mol_weight", "logP", "h_donors", "h_acceptors",
        "rot_bonds", "polar_area", "binding_affinity", "bioavailability",
        "half_life", "toxicity", "solubility", "drug_class",
        "path_inflammation", "path_metabolism", "path_apoptosis", "path_angiogenesis",
        "path_immunity", "genetic_risk", "severity", "prevalence",
        "age_onset", "comorbidity"
    ])

    df["drug_name"]    = [DRUG_NAMES[i]    for i in drug_indices]
    df["disease_name"] = [DISEASE_NAMES[i] for i in disease_indices]
    df["interaction"]  = y   # 0 = no repurposing potential, 1 = repurposing candidate

    # ── Known repurposing pairs (ground truth for validation) ──────────────
    # These are real scientifically validated repurposing cases
    known_pairs = [
        ("Metformin",  "Breast Cancer"),
        ("Metformin",  "Colorectal Cancer"),
        ("Aspirin",    "Colorectal Cancer"),
        ("Sildenafil", "Coronary Artery Disease"),
        ("Imatinib",   "Leukemia"),
        ("Rituximab",  "Rheumatoid Arthritis"),
        ("Adalimumab", "Psoriasis"),
        ("Methotrexate","Rheumatoid Arthritis"),
    ]

    for drug, disease in known_pairs:
        mask = (df["drug_name"] == drug) & (df["disease_name"] == disease)
        if mask.sum() > 0:
            df.loc[mask, "interaction"] = 1
        else:
            # Inject the known pair if not present
            row_idx = len(df)
            # Use mean features as placeholder for injected pairs
            mean_row = df.drop(columns=["drug_name","disease_name","interaction"]).mean()
            new_row  = mean_row.copy()
            new_row["drug_name"]    = drug
            new_row["disease_name"] = disease
            new_row["interaction"]  = 1
            df.loc[row_idx] = new_row

    df = df.reset_index(drop=True)

    print(f"✅ Dataset generated: {len(df)} samples")
    print(f"   Positive interactions (repurposing candidates): {df['interaction'].sum()} ({df['interaction'].mean()*100:.1f}%)")
    print(f"   Drugs: {df['drug_name'].nunique()} | Diseases: {df['disease_name'].nunique()}")
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_interaction_dataset(n_samples=2000)
    df.to_csv("data/drug_disease_interactions.csv", index=False)
    print("\n📁 Saved to data/drug_disease_interactions.csv")
    print(df.head())