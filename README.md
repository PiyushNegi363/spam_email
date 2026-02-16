# Spam Email Classification System

A production-ready spam/ham classifier built with Python, scikit-learn, and Streamlit. The app focuses on single-email classification, while the training pipeline supports model comparison and versioned outputs.

## What This Project Does

- Classifies email text as Spam or Ham with confidence scores.
- Trains multiple ML models and selects the best model based on cross-validation.
- Stores model artifacts and evaluation reports for each training run.

## How It Works

1. **Preprocessing**: Clean text (HTML removal, normalization, control-char stripping).
2. **Vectorization**: TF-IDF feature extraction.
3. **Modeling**: Train and compare models (e.g., SVM, Logistic Regression, Random Forest).
4. **Selection**: Choose the best model based on F1-score.
5. **Inference**: Load the best model and classify new emails in the Streamlit app.

## Key Features

- Streamlit UI for instant single-email classification.
- Modular pipeline for ingestion, transformation, training, and prediction.
- Detailed metrics and comparison reports saved per training run.
- Config-driven paths for datasets and model artifacts.

## Project Layout (Short)

```
app.py
src/
  components/         # ingestion, transformation, training
  pipeline/           # training & prediction pipelines
  utils/              # logging, text utilities
  config/             # project configuration
data/                 # datasets
outputs/              # model artifacts + reports per run
logs/                 # training/prediction logs
```

## Setup

```bash
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Paste an email and click **Classify Email** to see the prediction and confidence.

## Train a New Model

1. Prepare a CSV dataset with columns:

```csv
text,label
"your email text",spam
"another email",ham
```

2. Replace the dataset at:

```
data/dataset/dataset.csv
```

3. Run training:

```bash
python -m src.pipeline.training_pipeline
```

4. Update model paths in:

```
src/config/config.py
```

## Outputs and Logs

- Training artifacts and reports are stored in timestamped folders under:

```
outputs/
```

- Logs are written under:

```
logs/
```

## Configuration

The key paths are in:

```
src/config/config.py
```

Update `model_path` and `feature_path` after training if you want the app to use the latest model.

## Notes

- The app loads a saved model and vectorizer from `outputs/` using the paths in config.
- If scikit-learn versions differ between training and inference, re-train or install the matching version.
