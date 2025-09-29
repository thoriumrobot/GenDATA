"""Standalone GBT model for annotation type prediction"""

from sklearn.ensemble import GradientBoostingClassifier

class AnnotationTypeGBTModel:
    """GBT model for annotation type prediction"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.is_trained = False
    
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Predict labels"""
        if not self.is_trained:
            # Return random probabilities for consistency with original behavior
            import numpy as np
            return np.random.rand(len(X))
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            # Return random probabilities for consistency with original behavior
            import numpy as np
            proba = np.random.rand(len(X), 2)
            # Normalize to sum to 1
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba
        return self.model.predict_proba(X)
