from abc import ABC, abstractmethod
import joblib


class BaseModel(ABC):
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train, y_train, class_weights=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def save(self, filepath):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

