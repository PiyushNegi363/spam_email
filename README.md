# 📧 Spam Email Classifier Pro

A production-grade Machine Learning application to classify emails as **Spam** or **Ham** (Legitimate) with high precision. Built with a robust modular pipeline, premium Streamlit UI, and advanced error handling.

## 🌟 Key Features

- **🔍 Smart Classifier**: Analyze single emails using a high-precision Linear SVM model.
- **🚀 Batch Processing**: Upload CSV files for rapid, high-performance classification of thousands of emails.
- **📊 Model Insights**: Interactive dashboard displaying Accuracy, F1-Score, and Precision metrics.
- **🔄 Live Re-training**: Trigger the full ML pipeline (Ingestion -> Grid Search -> Serialization) directly from the UI.
- **🛡️ Reliability Audit Completed**: Hardened against crashes, file leaks, and malformed data.
- **🎨 Premium UI/UX**: Modern interface with custom typography, sidebar examples, and real-time feedback.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-Learn (SVM, Logistic Regression, Random Forest)
- **Data Processing**: Pandas, Numpy, TF-IDF Vectorization
- **State Management**: Python Dataclasses

## 📂 Project Structure

```text
├── app.py                # Main Streamlit Application
├── src/
│   ├── components/       # ML Components (Ingestion, Transformation, Training)
│   ├── pipeline/         # Execution Pipelines (Training, Prediction)
│   ├── config/           # Configuration Management
│   └── utils/            # Helper Utilities (Logger, Email Cleaning)
├── data/
│   └── dataset/          # Training Data
├── outputs/              # Trained Models & Evaluation Reports (Ignored by Git)
├── logs/                 # System Logs (Ignored by Git)
└── requirements.txt      # Project Dependencies
```

## 🚀 Quick Start

### 1. Setup Environment
```powershell
python -m venv .myenv
.myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Application
```powershell
streamlit run app.py
```

### 3. Re-train the Model
Navigate to the **Model Insights** tab in the UI and click **🔄 Retrain Model** to run the full training pipeline on the latest data.

## 📈 Performance
The currently deployed **SVM** model achieves:
- **Accuracy**: 97.9%
- **F1-Score**: 97.8%
- **Precision**: 97.9%

## 📝 License
MIT License. See `LICENSE` for details.
