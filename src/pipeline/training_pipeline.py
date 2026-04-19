from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.utils.state import TrainingState
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TrainingPipeline:
    """Complete training pipeline for spam classification"""
    
    def __init__(self):
        self.state = TrainingState()
        
    def run_pipeline(self, cv_folds: int = 5):
        try:
            import os
            from src.config.config import Config
            config = Config()
            
            logger.info(f"{'='*70}")
            logger.info("INITIATING TRAINING PIPELINE")
            logger.info(f"{'='*70}")
            
            # Pre-check: Dataset existence
            if not os.path.exists(config.training_data_path):
                logger.error(f"Dataset not found at: {config.training_data_path}")
                raise FileNotFoundError(f"Missing training data: {config.training_data_path}")

            # 1. Ingestion
            ingestion = DataIngestion()
            self.state = ingestion.load_data(self.state)
            
            # 2. Transformation
            transformation = DataTransformation()
            self.state = transformation.transform_data(self.state)
            
            # 3. Model Training
            trainer = ModelTraining()
            self.state = trainer.train_models(
                self.state, 
                cv_folds=cv_folds
            )
            
            # 4. Final Summary
            logger.info("\n" + "*"*70)
            logger.info("TRAINING PIPELINE SUMMARY")
            logger.info("*"*70)
            logger.info(f"Status: SUCCESS")
            logger.info(f"Best Model: {self.state.best_model_name}")
            
            metrics = self.state.model_metrics[self.state.best_model_name]
            logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"CV Score:  {metrics.get('best_cv_score', 0.0):.4f}")
            logger.info("*"*70)
            
            return self.state
            
        except Exception as e:
            logger.error(f"PIPELINE FAILED: {str(e)}", exc_info=True)
            raise e

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline(cv_folds=5)