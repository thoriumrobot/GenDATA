#!/usr/bin/env python3
"""
Filtered Multi-Checker Predictor

Extends MultiCheckerPredictor to filter predictions by base model.
Only predictions from the specified base model are considered.
"""

import logging
from typing import Optional
from multi_checker_predictor import MultiCheckerPredictor

logger = logging.getLogger(__name__)


class FilteredMultiCheckerPredictor(MultiCheckerPredictor):
    """
    Multi-Checker Predictor filtered to a specific base model.
    
    Only uses models from the specified base model for predictions,
    while still considering all annotation types for that base model.
    """
    
    def __init__(self, checker_name: str, base_model_filter: str, 
                 models_dir: Optional[str] = None, device: str = 'auto'):
        """
        Initialize filtered predictor.
        
        Args:
            checker_name: Name of the checker ('lower_bound', 'sql_quotes', 'signature_string')
            base_model_filter: Base model to filter by (e.g., 'gcn', 'hgt', 'gbt')
            models_dir: Directory containing models (defaults to checker-specific directory)
            device: Device to use ('auto', 'cuda', or 'cpu')
        """
        super().__init__(checker_name, models_dir, device)
        self.base_model_filter = base_model_filter.lower()
        
        logger.info(f"Initialized FilteredMultiCheckerPredictor with base_model_filter={base_model_filter}")
    
    def predict_for_location(self, cfg_data: dict, node: dict, 
                             line_number: int, threshold: float = 0.3) -> Optional[dict]:
        """
        Predict annotation for a single location using only the filtered base model.
        Returns the prediction with highest confidence if any model predicts "yes".
        
        Args:
            cfg_data: CFG data dictionary
            node: CFG node dictionary
            line_number: Line number of the location
            threshold: Confidence threshold for positive predictions
            
        Returns:
            Dict with keys: annotation_type, confidence, model_type, reason, line_number
            None if no models predict "yes"
        """
        if not self.loaded_models:
            logger.warning("No models loaded. Call load_checker_models() first.")
            return None
        
        # Extract features once
        try:
            features = self._extract_features(node, cfg_data, self.checker_name)
        except Exception as e:
            logger.debug(f"Failed to extract features: {e}")
            return None
        
        predictions = []
        
        # Run all annotation type models, but only for the filtered base model
        for annotation_type in self.annotation_types:
            if annotation_type not in self.loaded_models:
                continue
            
            # Only consider the filtered base model
            if self.base_model_filter not in self.loaded_models[annotation_type]:
                logger.debug(f"Base model {self.base_model_filter} not available for {annotation_type}")
                continue
            
            model_info = self.loaded_models[annotation_type][self.base_model_filter]
            
            # Get prediction (yes/no, confidence)
            is_positive, confidence, reason = self._get_model_prediction(
                model_info, features, annotation_type, self.base_model_filter
            )
            
            if is_positive and confidence >= threshold:
                predictions.append({
                    'annotation_type': annotation_type,
                    'confidence': confidence,
                    'model_type': self.base_model_filter,
                    'reason': reason,
                    'line_number': line_number
                })
        
        # Select highest confidence prediction
        if predictions:
            best_prediction = max(predictions, key=lambda p: p['confidence'])
            logger.debug(f"Selected {best_prediction['annotation_type']} (confidence: {best_prediction['confidence']:.3f}) "
                        f"from {len(predictions)} positive predictions at line {line_number} using {self.base_model_filter}")
            return best_prediction
        
        return None

