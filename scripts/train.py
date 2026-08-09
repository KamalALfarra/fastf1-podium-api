import comet_ml, json, joblib, argparse
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os
import pandas as pd

# NEW: Import metrics
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

# 1. Setup Recipes
RECIPES = {
    "1.0.0": ("Probabilistic RandomForest with class_weight={False: 1, True: 7}, max_iter=800",
              RandomForestClassifier(n_estimators=800, random_state=42,class_weight={False: 1, True: 7})),
    "1.1.0": ("Probabilistic LogisticRegression with class_weight={False: 1, True: 2}, max_iter=600",
              LogisticRegression(max_iter=600, random_state=42,class_weight={False: 1, True: 2})),
    "1.2.0": ("Probabilistic XGBoost with scale_pos_weight=8 eval_metric = aucpr,n_estimators=400,max_depth=2",
              XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=2, random_state=42,scale_pos_weight=8,eval_metric="aucpr")),
}

# 2. Parse Arguments
ap = argparse.ArgumentParser()
ap.add_argument("version", choices=sorted(RECIPES))
ap.add_argument("--promote", action="store_true", help="make this the live version")
args = ap.parse_args()

# 3. Initialize Comet Experiment
try:
    api_key = os.getenv("COMET_API_KEY")
    experiment = comet_ml.Experiment(
        api_key=api_key,
        project_name="first_F1_test",
        auto_metric_logging=True,
        auto_param_logging=True
    )
    experiment.log_parameter("version", args.version)
except Exception as e:
    print(f"Comet initialization skipped: {e}")
    experiment = None

# 4. Data & Training
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "final_features.csv")

note, estimator = RECIPES[args.version]
y = df['got_podium']
X = df.drop(columns=['got_podium', 'constructor_name', 'driver_name'])

train_mask = df["season"] <= 2021
val_mask = df["season"].isin([2022, 2023])
test_mask = df["season"] == 2024

Xtr = X[train_mask]
ytr = y[train_mask]

Xval = X[val_mask]
yval = y[val_mask]

Xte = X[test_mask]
yte = y[test_mask]

print(f"Training rows:   {len(Xtr)}")
print(f"Validation rows: {len(Xval)}")
print(f"Test rows:       {len(Xte)}")

print(f"Training races:   {df.loc[train_mask, ['season','round']].drop_duplicates().shape[0]}")
print(f"Validation races: {df.loc[val_mask, ['season','round']].drop_duplicates().shape[0]}")
print(f"Test races:       {df.loc[test_mask, ['season','round']].drop_duplicates().shape[0]}")
pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)]).fit(Xtr, ytr)

# Generate predictions to calculate metrics
probs = pipe.predict_proba(Xte)[:, 1]
# Create evaluation dataframe
results = df.loc[Xte.index, ['season','round', 'driver_name', 'got_podium']].copy()
results['podium_probability'] = probs
# Rank drivers within each race
results['rank'] = (
    results
    .groupby(['season', 'round'])['podium_probability']
    .rank(method='first', ascending=False)
)

# Predict top 3 drivers for each race
results['predicted_podium'] = results['rank'] <= 3

# How many actual podium drivers were in our predicted top 3?
correct_podiums = (
    results[
        results['predicted_podium'] & results['got_podium']
    ]
    .groupby(['season', 'round'])
    .size()
)
# Average number of correct podium predictions per race
average_correct = correct_podiums.mean()

# Percentage of actual podium spots correctly predicted
podium_recall_at_3 = (
    results['predicted_podium'] & results['got_podium']
).sum() / results['got_podium'].sum()

# ============================================================
# Top-3 Podium Evaluation
# ============================================================

# Create evaluation dataframe
results = df.loc[
    Xte.index,
    ['season', 'round', 'driver_name', 'got_podium']
].copy()

# Add predicted podium probability
results['podium_probability'] = probs

# Rank drivers within each race
results['rank'] = (
    results
    .groupby(['season', 'round'])['podium_probability']
    .rank(method='first', ascending=False)
)

# Top 3 drivers are our predicted podium
results['predicted_podium'] = results['rank'] <= 3

# Count how many of our predicted top 3 actually podiumed
correct_predictions = (
    results['predicted_podium'] & results['got_podium']
).sum()

# Number of races in the test set
number_of_races = (
    results[['season', 'round']]
    .drop_duplicates()
    .shape[0]
)

# Average number of correct podium drivers per race
average_correct = correct_predictions / number_of_races

# Since every complete F1 race has exactly 3 podium positions:
podium_recall_at_3 = average_correct / 3

# Print results ONCE
print(f"\n--- Top 3 Podium Prediction for v{args.version} ---")
print(f"Average correct podiums per race: {average_correct:.2f} / 3")
print(f"Podium Recall@3: {podium_recall_at_3:.2%}")

# 5. Save Locally
out = Path("models") / f"v{args.version}"
out.mkdir(parents=True, exist_ok=True)
model_path = out / "model.joblib"
joblib.dump(pipe, model_path)

# UPDATED: Add new metrics to your local JSON
meta = {
    "model_version": args.version,
    "description": note,
    "algorithm": type(estimator).__name__,
    "features": list(X.columns),

    "test_average_correct_podiums": round(
        float(average_correct), 4
    ),

    "test_recall_at_3": round(
        float(podium_recall_at_3), 4
    ),

    "test_races": int(number_of_races),
    "test_rows": int(len(Xte)),
}

(out / "meta.json").write_text(json.dumps(meta, indent=2))

# 6. Log to Comet
if experiment:
    experiment.log_parameters(meta)

    # UPDATED: Log the expanded dictionary of metrics
    if experiment:
        experiment.log_parameters(meta)

        experiment.log_metrics({
            "test_average_correct_podiums": meta["test_average_correct_podiums"],
            "test_recall_at_3": meta["test_recall_at_3"],
        })

        experiment.set_name(note)

        experiment.log_model(
            f"f1-model-v{args.version}",
            str(model_path)
        )

        experiment.end()

print(f"Saved {out} | {note}")
if args.promote:
    Path("models/CURRENT").write_text(f"v{args.version}\n")
    print(f"CURRENT -> v{args.version} (Live)")