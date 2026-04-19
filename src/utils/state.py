from typing import Optional, List, Dict, Any
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class TrainingState:
    training_data_path: Optional[str] = None
    training_data: Optional[pd.DataFrame] = None
    transformed_data: Optional[pd.DataFrame] = None
    X_train: Optional[pd.Series] = None
    X_test: Optional[pd.Series] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    X_train_tfidf: Optional[Any] = None
    X_test_tfidf: Optional[Any] = None
    tfidf_vectorizer: Optional[Any] = None
    trained_models: Dict[str, Any] = field(default_factory=dict)
    model_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    best_model_name: Optional[str] = None
    best_model: Optional[Any] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    cv_results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionState:
    mailbox_path: Optional[str] = None
    mail_data: List[Dict[str, str]] = field(default_factory=list)