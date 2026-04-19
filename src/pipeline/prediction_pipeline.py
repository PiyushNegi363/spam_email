import os
import pickle
import numpy as np
from typing import Dict

from src.utils.logger import get_logger
from src.config.config import Config
from src.utils.email_utils import clean_text

logger = get_logger(__name__)

class PredictionPipeline:
    def __init__(self, load_models: bool = True):
        self.config = Config()
        self.feature_transformer = None
        self.model = None
        
        if load_models:
            self._load_models()
    
    def _load_models(self) -> None:
        try:
            logger.info(f"Loading feature transformer from: {self.config.feature_path}")
            if not os.path.exists(self.config.feature_path):
                raise FileNotFoundError(f"Feature transformer not found at {self.config.feature_path}")
            
            with open(self.config.feature_path, "rb") as f:
                self.feature_transformer = pickle.load(f)
            
            logger.info(f"Loading model from: {self.config.model_path}")
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model not found at {self.config.model_path}")
                
            with open(self.config.model_path, "rb") as f:
                self.model = pickle.load(f)
                
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise e
    
    def predict_single_email(self, email_body: str) -> Dict:
        if self.model is None or self.feature_transformer is None:
            self._load_models()

        if not email_body.strip():
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'raw_prediction': -1,
                'error': 'Empty email body'
            }

        cleaned_body = clean_text(email_body)
        features = self.feature_transformer.transform([cleaned_body])
        prediction = self.model.predict(features)
        
        # Robust label mapping
        pred_val = str(prediction[0])
        prediction_label = "Spam" if pred_val in ["0", "spam"] else "Ham"
        
        confidence = 0.0
        try:
            if hasattr(self.model, "predict_proba"):
                prediction_proba = self.model.predict_proba(features)
                confidence = float(np.max(prediction_proba[0])) * 100
            elif hasattr(self.model, "decision_function"):
                confidence = 0.0 
        except Exception as e:
            logger.warning(f"Could not calculate confidence: {str(e)}")
            confidence = 0.0
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'raw_prediction': int(prediction[0])
        }

    def predict_batch(self, email_bodies: list) -> list:
        """Process multiple emails for high-performance batch classification"""
        if self.model is None or self.feature_transformer is None:
            self._load_models()

        if not email_bodies:
            return []

        # Clean all texts
        cleaned_bodies = [clean_text(text) for text in email_bodies]
        
        # Vectorize all at once
        features = self.feature_transformer.transform(cleaned_bodies)
        
        # Predict all at once
        predictions = self.model.predict(features)
        
        results = []
        
        # Get probabilities if available
        probas = None
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(features)

        for i, pred in enumerate(predictions):
            pred_val = str(pred)
            label = "Spam" if pred_val in ["0", "spam"] else "Ham"
            
            conf = 0.0
            if probas is not None:
                conf = float(np.max(probas[i])) * 100
                
            results.append({
                'text': email_bodies[i],
                'prediction': label,
                'confidence': conf
            })
            
        return results
