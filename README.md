# F1 Podium Predictor 🏎️🏆

> **F1 Project**: This project develops a machine-learning model that predicts whether an F1 driver will finish on the podium in a given race. On the races evaluated, the model correctly identified approximately two out of every three podium finishers.

![F1 API](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![XGBoost](https://img.shields.io/badge/XGBoost-1793D1?style=for-the-badge&logo=xgboost) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Comet](https://img.shields.io/badge/Comet_ML-111111?style=for-the-badge)

---

## 📌 Project Overview

This project is a full-stack Machine Learning application built for a capstone project. It integrates a robust data engineering pipeline, a machine learning training script with experiment tracking, a FastAPI-based model serving backend, and a dynamic, beautiful user interface.

The core objective is to predict race outcomes by computing a **Podium Probability** for any driver, leveraging historical data, career statistics, and current season performance.

## 📂 Project Architecture

```text
fastf1-podium-api/
│
├── app/
│   └── main.py              # FastAPI server and inference logic
├── data/
│   ├── raw/                 # Raw JSON/CSV dumps from the Ergast API
│   └── processed/           # Final merged feature sets for ML training
├── models/                  # Versioned ML models and metadata
│   ├── CURRENT              # Pointer to the currently active live model
│   └── v1.2.0/              # Version-specific model artifacts (model.joblib, meta.json)
├── scripts/
│   ├── get_data.py          # Data extraction and feature engineering pipeline
│   ├── train.py             # Model training, evaluation, and logging script
│   └── baseline.py          # Baseline model for comparison
├── ui.html                  # Premium frontend User Interface
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation (You are here)
```

---

## ⚙️ The Data Pipeline (`scripts/get_data.py`)

The data pipeline extracts historical Formula 1 data from the **Jolpi Ergast F1 API**. 

### 1. Extraction & Feature Engineering
- **Historical Data**: Fetches all races, race results, driver standings, and constructor standings from 2014 to 2024.
- **Career Features**: Computes career statistics dynamically for each driver (e.g., `career_wins_before_race`, `career_podiums_before_race`, `years_of_experience`, `age_at_race`).
- **Recent Form**: Calculates a rolling average of a driver's finishing positions over their last 3 races (`driver_recent_form`).
- **Data Leakage Prevention**: Strictly utilizes **PRE-RACE** championship standings and points to ensure the model doesn't peak into the future. 

### 2. Merging & Imputation
- The pipeline merges race context, driver history, and constructor performance into a single unified dataset.
- Missing values (like grid positions for pit-lane starts) are safely imputed and flagged (`grid_position_missing`).
- The target variable `got_podium` is created as a boolean indicator. The final dataset is saved to `data/processed/final_features.csv`.

---

## 🧠 Machine Learning Model (`scripts/train.py`)

The project supports multiple algorithmic recipes (Random Forest, Logistic Regression, XGBoost). The live model defaults to **XGBoost (v1.2.0)** due to its ability to handle non-linear relationships and class imbalances.

### Training Strategy
- **Chronological Split**: Time-series based splitting to simulate real-world prediction.
  - **Training Set**: Seasons <= 2021
  - **Validation Set**: Seasons 2022 - 2023
  - **Testing Set**: Season 2024
- **Class Imbalance**: Formula 1 is highly imbalanced (only 3 podium spots out of ~20 drivers). The models utilize parameters like `scale_pos_weight` and `class_weight` to emphasize podium predictions.

### Evaluation Metrics
Instead of standard accuracy, the model calculates **Podium Recall@3**. 
For every race in the test set, the model outputs probabilities for all drivers, ranks them, and selects the top 3. We then evaluate how many of the actual real-life podium drivers were inside our predicted Top 3.

### Experiment Tracking
The script integrates with **Comet ML** via the `COMET_API_KEY` environment variable. It automatically logs hyper-parameters, datasets, metrics (`test_recall_at_3`), and saves the trained `.joblib` model pipeline directly to the cloud and local `models/` directory.

---

## 🚀 Serving Backend (`app/main.py`)

The application is served using **FastAPI**.
- **Dynamic Loading**: On startup (`lifespan`), the server reads the `models/CURRENT` file, identifying the active version, and loads the corresponding `.joblib` model and `meta.json` strictly once into memory.
- **Strict Enforcement**: The `/predict` endpoint uses Pydantic to accept JSON payloads. It parses the incoming features and leverages `pandas` to align the payload columns *exactly* to the `meta["features"]` array generated during training.
- **Endpoints**:
  - `GET /`: Serves the `ui.html` frontend.
  - `POST /predict`: Takes driver/race context and returns the boolean prediction and `podium_probability`.
  - `GET /health` & `GET /metadata`: Expose model status and feature dependencies.

---

## 🎨 User Interface (`ui.html`)

A highly polished, dark-themed, premium frontend designed with HTML, CSS, and Vanilla JavaScript. 
- **Aesthetics**: Glassmorphism translucent cards, dynamic hover micro-animations, and F1 red accent colors.
- **Dynamic Visualization**: Submitting a prediction asynchronously pings the FastAPI backend and smoothly renders a circular probability graph and a stylized "PODIUM" or "NO PODIUM" banner based on the model's output probability.

---

## 🛠️ Installation & Usage

### 1. Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd fastf1-podium-api

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train
```bash
# Fetch data from Ergast API and build features
python scripts/get_data.py

# Train the model (e.g., version 1.2.0) and promote it to CURRENT
python scripts/train.py 1.2.0 --promote
```

### 3. Run the Application
```bash
# Start the FastAPI server
uvicorn app.main:app --reload
```
Once the server is running, open your web browser and navigate to **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** to interact with the UI.

---
*Developed as a Capstone Project.*
