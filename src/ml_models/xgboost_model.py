try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

from .base_model import BaseModel
import numpy as np


class XGBoostModel(BaseModel):
    
    def __init__(self):
        super().__init__()
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    
    def train(self, X_train, y_train, class_weights=None):
        sample_weight = None
        if class_weights is not None:
            sample_weight = np.array([class_weights[y] for y in y_train])
        
        self.model.fit(
            X_train, 
            y_train,
            sample_weight=sample_weight
        )
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

