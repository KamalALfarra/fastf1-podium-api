import comet_ml, json, joblib, argparse
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

# NEW: Import metrics
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

# 1. Setup Recipes
RECIPES = {
    "1.0.0": ("RandomForest, 100 trees, untuned",
              RandomForestClassifier(n_estimators=100, random_state=42,class_weight={False: 1, True: 30})),
    "1.1.0": ("LogisticRegression, better-calibrated probabilities",
              LogisticRegression(max_iter=500, random_state=42,class_weight={False: 1, True: 10})),
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
df["grid_position_missing"] = df["grid_position"].isna().astype(int)
df["grid_position"] = df["grid_position"].fillna(24)

note, estimator = RECIPES[args.version]
y = df['got_podium']
X = df.drop(columns=['got_podium', 'constructor_name', 'driver_name'])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)]).fit(Xtr, ytr)

# Generate predictions to calculate metrics
preds = pipe.predict(Xte)

# Print the classification report to your terminal
print(f"\n--- Classification Report for v{args.version} ---")
print(classification_report(yte, preds))

# NEW: Calculate individual metrics (using 'weighted' to handle potential class imbalances)
test_precision = precision_score(yte, preds, pos_label=True, zero_division=0)
test_recall = recall_score(yte, preds, pos_label=True, zero_division=0)
test_f1 = f1_score(yte, preds, pos_label=True, zero_division=0)


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
    "test_accuracy": round(float(pipe.score(Xte, yte)), 4),
    "test_precision": round(float(test_precision), 4),
    "test_recall": round(float(test_recall), 4),
    "test_f1": round(float(test_f1), 4),
    "cv_accuracy": round(float(cross_val_score(pipe, Xtr, ytr, cv=5).mean()), 4),
}
(out / "meta.json").write_text(json.dumps(meta, indent=2))

# 6. Log to Comet
if experiment:
    experiment.log_parameters(meta)

    # UPDATED: Log the expanded dictionary of metrics
    experiment.log_metrics({
        "test_acc": meta["test_accuracy"],
        "cv_acc": meta["cv_accuracy"],
        "test_precision": meta["test_precision"],
        "test_recall": meta["test_recall"],
        "test_f1": meta["test_f1"]
    })

    # Log a visual confusion matrix directly to Comet!
    experiment.log_confusion_matrix(y_true=yte, y_predicted=preds)
    experiment.set_name("RandomForest with class_weight={False: 1, True: 30}")

    experiment.log_model(f"f1-model-v{args.version}", str(model_path))
    experiment.end()

print(f"Saved {out} | {note}")
if args.promote:
    Path("models/CURRENT").write_text(f"v{args.version}\n")
    print(f"CURRENT -> v{args.version} (Live)")