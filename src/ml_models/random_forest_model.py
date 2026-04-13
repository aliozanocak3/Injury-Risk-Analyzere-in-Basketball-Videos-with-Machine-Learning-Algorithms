from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    def train(self, X_train, y_train, class_weights=None):
        if class_weights is not None:
            self.model.set_params(class_weight=class_weights)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

