from src.utils.logger import get_logger
from src.config.config import Config
from src.utils.state import TrainingState
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self):
        self.config = Config()
    
    def transform_data(self, state: TrainingState) -> TrainingState:
        logger.info("Data transformation started")
        try:
            data = state.training_data.copy()
            
            # 1. Handle Missing Values
            # ------------------------------------------------------------------
            initial_rows = len(data)
            data.dropna(subset=['Category', 'Message'], inplace=True)
            if len(data) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(data)} rows with missing values")
            
            # 2. Robust Label Encoding
            # ------------------------------------------------------------------
            # Convert to string and strip/lowercase for matching
            data['Category'] = data['Category'].astype(str).str.strip().str.lower()
            
            # Mapping: spam -> 0, ham -> 1. 
            # We use a explicit map to handle unexpected values
            label_map = {'spam': 0, 'ham': 1}
            
            # Filter to only rows we can handle, or map everything else to ham? 
            # Better to drop unknown categories to ensure training quality.
            valid_mask = data['Category'].isin(label_map.keys())
            if not valid_mask.all():
                logger.warning(f"Dropping {len(data) - valid_mask.sum()} rows with unknown categories: {data.loc[~valid_mask, 'Category'].unique()}")
                data = data[valid_mask]
            
            data['Category'] = data['Category'].map(label_map).astype(int)
            
            logger.info(f"Label encoding completed. Data shape: {data.shape}")
            logger.info(f"Unique labels: {data['Category'].unique()}")
            
            # 3. Text Preprocessing
            # ------------------------------------------------------------------
            # Ensure Message is string
            data['Message'] = data['Message'].astype(str)
            
            # 4. Split and Vectorize
            # ------------------------------------------------------------------
            X = data['Message']
            y = data['Category'].values
            
            # Split into train and test sets (70:30 ratio)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            logger.info(f"Train/test split completed. Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Apply TF-IDF vectorization
            tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', min_df=2)
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)
            
            logger.info(f"TF-IDF transformation completed. Feature shape: {X_train_tfidf.shape}")
            
            # Save to state
            state.transformed_data = data
            state.X_train = X_train
            state.X_test = X_test
            state.y_train = y_train
            state.y_test = y_test
            state.X_train_tfidf = X_train_tfidf
            state.X_test_tfidf = X_test_tfidf
            state.tfidf_vectorizer = tfidf_vectorizer
            
            logger.info("Data transformation completed successfully")
            return state
        except Exception as e:
            logger.error(f"Failed to transform data: {str(e)}")
            raise e