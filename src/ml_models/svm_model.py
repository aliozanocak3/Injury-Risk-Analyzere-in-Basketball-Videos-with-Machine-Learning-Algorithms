from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
import numpy as np


class SVMModel(BaseModel):
    
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.model = SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=42,
            probability=True
        )
    
    def train(self, X_train, y_train, class_weights=None):
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if class_weights is not None:
            self.model.set_params(class_weight=class_weights)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

