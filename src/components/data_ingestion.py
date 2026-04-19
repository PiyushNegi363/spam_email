import pandas as pd
from src.utils.logger import get_logger
from src.config.config import Config
from src.utils.state import TrainingState

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.config = Config()
    
    def load_data(self, state: TrainingState) -> TrainingState:
        try:
            logger.info(f"Loading data from: {self.config.training_data_path}")
            
            import os
            if not os.path.exists(self.config.training_data_path):
                raise FileNotFoundError(f"Dataset not found at {self.config.training_data_path}")
                
            df = pd.read_csv(self.config.training_data_path)
            
            # Validation
            required_columns = ['Category', 'Message']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Dataset missing required columns: {missing_cols}")
            
            if df.empty:
                raise ValueError("Dataset is empty")
                
            state.training_data = df
            logger.info(f"Data loaded successfully. Rows: {len(df)}")
            return state
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise e
    