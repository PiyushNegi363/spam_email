import pickle
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

        logger.info("Loading models...")
        self.feature_transformer = pickle.load(open(self.config.feature_path, "rb"))
        self.model = pickle.load(open(self.config.model_path, "rb"))
        logger.info("Models loaded successfully")
    
    def predict_single_email(self, email_body: str) -> Dict:
        if self.model is None or self.feature_transformer is None:
            self._load_models()

        cleaned_body = clean_text(email_body)
        features = self.feature_transformer.transform([cleaned_body])
        prediction = self.model.predict(features)
        prediction_label = "Spam" if str(prediction[0]) == "0" else "Ham"
        
        try:
            prediction_proba = self.model.predict_proba(features)
            confidence = float(max(prediction_proba[0])) * 100
        except:
            confidence = None
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'raw_prediction': int(prediction[0])
        }
