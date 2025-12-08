#!/usr/bin/env python3
"""
Checker-Specific Model Architectures

Separate model architectures per checker that integrate checker value attention
with base model types (GCN, HGT, GBT, Causal, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from checker_config import CheckerType, get_checker_config
from checker_value_attention import CheckerValueAttention, CheckerValueAttentionPool
from value_pattern_detector import ValuePatternDetector

# Import base model classes
try:
    from graph_based_annotation_models import GraphBasedAnnotationModel, GraphBasedGCNModel, GraphBasedHGTModel
    GRAPH_MODELS_AVAILABLE = True
except ImportError:
    GRAPH_MODELS_AVAILABLE = False

try:
    from enhanced_causal_model import EnhancedCausalModel
    ENHANCED_CAUSAL_AVAILABLE = True
except ImportError:
    ENHANCED_CAUSAL_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier


class CheckerSpecificBaseModel(nn.Module):
    """Base class for checker-specific models"""
    
    def __init__(
        self,
        checker_type: CheckerType,
        base_model_type: str,
        input_dim: int,
        pattern_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 2,
        **kwargs
    ):
        super().__init__()
        self.checker_type = checker_type
        self.base_model_type = base_model_type
        self.input_dim = input_dim
        self.pattern_dim = pattern_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        config = get_checker_config(checker_type)
        self.checker_name = config.get('name', 'Unknown')
        
        # Value pattern detector
        self.pattern_detector = ValuePatternDetector()
        
        # Checker-specific value attention
        self.value_attention = CheckerValueAttention(
            checker_type=checker_type,
            pattern_dim=pattern_dim,
            hidden_dim=hidden_dim // 2,
            num_attention_heads=4
        )
    
    def extract_value_patterns(self, node: Dict, cfg_data: Dict) -> torch.Tensor:
        """Extract and emphasize value patterns for this checker"""
        patterns = self.pattern_detector.detect_patterns(node, cfg_data, self.checker_type)
        pattern_list = self.pattern_detector.get_pattern_features(node, cfg_data, self.checker_type)
        pattern_tensor = torch.tensor(pattern_list, dtype=torch.float32)
        
        # Apply learned attention emphasis
        emphasized = self.value_attention(pattern_tensor)
        return emphasized
    
    def get_attention_summary(self) -> Dict[str, float]:
        """Get summary of learned emphasis weights"""
        return self.value_attention.get_attention_summary(
            torch.zeros(self.pattern_dim)
        )


class LowerBoundCheckerModel(CheckerSpecificBaseModel):
    """Model for Lower Bound Checker with learned 0/-1 emphasis"""
    
    def __init__(self, base_model_type: str, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, **kwargs):
        # Lower Bound Checker has 10 value patterns
        pattern_dim = 10
        super().__init__(
            checker_type=CheckerType.INDEX,
            base_model_type=base_model_type,
            input_dim=input_dim,
            pattern_dim=pattern_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            **kwargs
        )
        
        # Initialize base model
        self.base_model = self._create_base_model(input_dim + pattern_dim, hidden_dim, out_dim, **kwargs)
    
    def _create_base_model(self, input_dim: int, hidden_dim: int, out_dim: int, **kwargs):
        """Create base model based on model type"""
        if self.base_model_type == 'gbt':
            # GBT doesn't use PyTorch, handled separately
            return None
        elif self.base_model_type in ['gcn', 'hgt', 'gcsn']:
            # Graph-based models - will be handled in forward
            return None
        elif self.base_model_type == 'causal':
            from graph_causal_model import GraphCausalModel
            return GraphCausalModel(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, **kwargs)
        elif self.base_model_type == 'enhanced_causal':
            if ENHANCED_CAUSAL_AVAILABLE:
                return EnhancedCausalModel(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, **kwargs)
            else:
                raise ImportError("Enhanced causal model not available")
        else:
            # Default: simple MLP
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, out_dim)
            )
    
    def forward(self, x: torch.Tensor, pattern_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with value pattern emphasis"""
        if pattern_features is not None:
            # Apply attention to pattern features
            emphasized_patterns = self.value_attention(pattern_features)
            # Concatenate with base features
            x = torch.cat([x, emphasized_patterns], dim=-1)
        
        if self.base_model is not None:
            return self.base_model(x)
        else:
            # For graph-based models, pattern features are integrated differently
            return x


class NullCheckerModel(CheckerSpecificBaseModel):
    """Model for Null Checker with learned null emphasis"""
    
    def __init__(self, base_model_type: str, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, **kwargs):
        # Null Checker has 8 value patterns
        pattern_dim = 8
        super().__init__(
            checker_type=CheckerType.NULLNESS,
            base_model_type=base_model_type,
            input_dim=input_dim,
            pattern_dim=pattern_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            **kwargs
        )
        
        self.base_model = self._create_base_model(input_dim + pattern_dim, hidden_dim, out_dim, **kwargs)
    
    def _create_base_model(self, input_dim: int, hidden_dim: int, out_dim: int, **kwargs):
        """Create base model - same as LowerBoundCheckerModel"""
        if self.base_model_type == 'gbt':
            return None
        elif self.base_model_type in ['gcn', 'hgt', 'gcsn']:
            return None
        elif self.base_model_type == 'causal':
            from graph_causal_model import GraphCausalModel
            return GraphCausalModel(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, **kwargs)
        elif self.base_model_type == 'enhanced_causal':
            if ENHANCED_CAUSAL_AVAILABLE:
                return EnhancedCausalModel(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, **kwargs)
            else:
                raise ImportError("Enhanced causal model not available")
        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, out_dim)
            )
    
    def forward(self, x: torch.Tensor, pattern_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with null pattern emphasis"""
        if pattern_features is not None:
            emphasized_patterns = self.value_attention(pattern_features)
            x = torch.cat([x, emphasized_patterns], dim=-1)
        
        if self.base_model is not None:
            return self.base_model(x)
        else:
            return x


class SignatureStringCheckerModel(CheckerSpecificBaseModel):
    """Model for Signature String Checker with learned string emphasis"""
    
    def __init__(self, base_model_type: str, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, **kwargs):
        # Signature String Checker has 8 value patterns
        pattern_dim = 8
        super().__init__(
            checker_type=CheckerType.SIGNATURE,
            base_model_type=base_model_type,
            input_dim=input_dim,
            pattern_dim=pattern_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            **kwargs
        )
        
        self.base_model = self._create_base_model(input_dim + pattern_dim, hidden_dim, out_dim, **kwargs)
    
    def _create_base_model(self, input_dim: int, hidden_dim: int, out_dim: int, **kwargs):
        """Create base model"""
        if self.base_model_type == 'gbt':
            return None
        elif self.base_model_type in ['gcn', 'hgt', 'gcsn']:
            return None
        elif self.base_model_type == 'causal':
            from graph_causal_model import GraphCausalModel
            return GraphCausalModel(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, **kwargs)
        elif self.base_model_type == 'enhanced_causal':
            if ENHANCED_CAUSAL_AVAILABLE:
                return EnhancedCausalModel(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, **kwargs)
            else:
                raise ImportError("Enhanced causal model not available")
        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, out_dim)
            )
    
    def forward(self, x: torch.Tensor, pattern_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with string pattern emphasis"""
        if pattern_features is not None:
            emphasized_patterns = self.value_attention(pattern_features)
            x = torch.cat([x, emphasized_patterns], dim=-1)
        
        if self.base_model is not None:
            return self.base_model(x)
        else:
            return x


class InterningCheckerModel(CheckerSpecificBaseModel):
    """Model for Interning Checker with learned interned string emphasis"""
    
    def __init__(self, base_model_type: str, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, **kwargs):
        pattern_dim = 6
        super().__init__(
            checker_type=CheckerType.INTERNING,
            base_model_type=base_model_type,
            input_dim=input_dim,
            pattern_dim=pattern_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            **kwargs
        )
        self.base_model = self._create_base_model(input_dim + pattern_dim, hidden_dim, out_dim, **kwargs)
    
    def _create_base_model(self, input_dim: int, hidden_dim: int, out_dim: int, **kwargs):
        """Create base model"""
        if self.base_model_type == 'gbt':
            return None
        elif self.base_model_type in ['gcn', 'hgt', 'gcsn']:
            return None
        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, out_dim)
            )
    
    def forward(self, x: torch.Tensor, pattern_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        if pattern_features is not None:
            emphasized_patterns = self.value_attention(pattern_features)
            x = torch.cat([x, emphasized_patterns], dim=-1)
        if self.base_model is not None:
            return self.base_model(x)
        return x


class LockCheckerModel(CheckerSpecificBaseModel):
    """Model for Lock Checker with learned lock emphasis"""
    
    def __init__(self, base_model_type: str, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, **kwargs):
        pattern_dim = 6
        super().__init__(
            checker_type=CheckerType.LOCK,
            base_model_type=base_model_type,
            input_dim=input_dim,
            pattern_dim=pattern_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            **kwargs
        )
        self.base_model = self._create_base_model(input_dim + pattern_dim, hidden_dim, out_dim, **kwargs)
    
    def _create_base_model(self, input_dim: int, hidden_dim: int, out_dim: int, **kwargs):
        """Create base model"""
        if self.base_model_type == 'gbt':
            return None
        elif self.base_model_type in ['gcn', 'hgt', 'gcsn']:
            return None
        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, out_dim)
            )
    
    def forward(self, x: torch.Tensor, pattern_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        if pattern_features is not None:
            emphasized_patterns = self.value_attention(pattern_features)
            x = torch.cat([x, emphasized_patterns], dim=-1)
        if self.base_model is not None:
            return self.base_model(x)
        return x


class RegexCheckerModel(CheckerSpecificBaseModel):
    """Model for Regex Checker with learned regex pattern emphasis"""
    
    def __init__(self, base_model_type: str, input_dim: int, hidden_dim: int = 128, out_dim: int = 2, **kwargs):
        pattern_dim = 6
        super().__init__(
            checker_type=CheckerType.REGEX,
            base_model_type=base_model_type,
            input_dim=input_dim,
            pattern_dim=pattern_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            **kwargs
        )
        self.base_model = self._create_base_model(input_dim + pattern_dim, hidden_dim, out_dim, **kwargs)
    
    def _create_base_model(self, input_dim: int, hidden_dim: int, out_dim: int, **kwargs):
        """Create base model"""
        if self.base_model_type == 'gbt':
            return None
        elif self.base_model_type in ['gcn', 'hgt', 'gcsn']:
            return None
        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, out_dim)
            )
    
    def forward(self, x: torch.Tensor, pattern_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        if pattern_features is not None:
            emphasized_patterns = self.value_attention(pattern_features)
            x = torch.cat([x, emphasized_patterns], dim=-1)
        if self.base_model is not None:
            return self.base_model(x)
        return x


def create_checker_specific_model(
    checker_type: CheckerType,
    base_model_type: str,
    input_dim: int,
    hidden_dim: int = 128,
    out_dim: int = 2,
    **kwargs
) -> CheckerSpecificBaseModel:
    """
    Factory function to create checker-specific model
    
    Args:
        checker_type: Type of checker
        base_model_type: Base model type (gcn, hgt, gbt, causal, etc.)
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        **kwargs: Additional arguments
        
    Returns:
        Checker-specific model instance
    """
    if checker_type == CheckerType.INDEX:
        return LowerBoundCheckerModel(base_model_type, input_dim, hidden_dim, out_dim, **kwargs)
    elif checker_type == CheckerType.NULLNESS:
        return NullCheckerModel(base_model_type, input_dim, hidden_dim, out_dim, **kwargs)
    elif checker_type == CheckerType.SIGNATURE:
        return SignatureStringCheckerModel(base_model_type, input_dim, hidden_dim, out_dim, **kwargs)
    elif checker_type == CheckerType.INTERNING:
        return InterningCheckerModel(base_model_type, input_dim, hidden_dim, out_dim, **kwargs)
    elif checker_type == CheckerType.LOCK:
        return LockCheckerModel(base_model_type, input_dim, hidden_dim, out_dim, **kwargs)
    elif checker_type == CheckerType.REGEX:
        return RegexCheckerModel(base_model_type, input_dim, hidden_dim, out_dim, **kwargs)
    else:
        raise ValueError(f"Unknown checker type: {checker_type}")

