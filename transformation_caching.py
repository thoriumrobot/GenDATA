#!/usr/bin/env python3
"""
Transformation Caching Module

This module implements caching mechanisms for model loading, transformation success patterns,
and performance optimization to improve the efficiency of the augmentation system.
"""

import os
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

from recursive_augmentation_engine import TransformationType, TransformationState

logger = logging.getLogger(__name__)

@dataclass
class TransformationResult:
    """Cached result of a transformation"""
    transformation_type: str
    input_hash: str
    output_code: str
    success: bool
    execution_time: float
    location_context: Dict[str, Any]
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class SuccessPattern:
    """Pattern of successful transformations"""
    code_pattern_hash: str
    successful_transformations: List[str]
    success_rates: Dict[str, float]
    location_patterns: Dict[str, List[str]]
    last_updated: str
    usage_count: int

class TransformationCache:
    """Cache for transformation results and success patterns"""
    
    def __init__(self, cache_dir: str = 'transformation_cache', max_size: int = 10000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache files
        self.results_cache_file = os.path.join(cache_dir, 'transformation_results.pkl')
        self.patterns_cache_file = os.path.join(cache_dir, 'success_patterns.pkl')
        self.model_cache_file = os.path.join(cache_dir, 'model_cache.pkl')
        
        # In-memory caches
        self.transformation_results: Dict[str, TransformationResult] = {}
        self.success_patterns: Dict[str, SuccessPattern] = {}
        self.model_cache: Dict[str, Any] = {}
        
        # Statistics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'load_time': 0.0,
            'save_time': 0.0
        }
        
        # Load existing caches
        self._load_caches()
    
    def _load_caches(self):
        """Load caches from disk"""
        start_time = datetime.now()
        
        try:
            # Load transformation results
            if os.path.exists(self.results_cache_file):
                with open(self.results_cache_file, 'rb') as f:
                    self.transformation_results = pickle.load(f)
                logger.info(f"Loaded {len(self.transformation_results)} transformation results from cache")
            
            # Load success patterns
            if os.path.exists(self.patterns_cache_file):
                with open(self.patterns_cache_file, 'rb') as f:
                    self.success_patterns = pickle.load(f)
                logger.info(f"Loaded {len(self.success_patterns)} success patterns from cache")
            
            # Load model cache
            if os.path.exists(self.model_cache_file):
                with open(self.model_cache_file, 'rb') as f:
                    self.model_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.model_cache)} models from cache")
            
        except Exception as e:
            logger.error(f"Error loading caches: {e}")
        
        self.cache_stats['load_time'] = (datetime.now() - start_time).total_seconds()
    
    def _save_caches(self):
        """Save caches to disk"""
        start_time = datetime.now()
        
        try:
            # Save transformation results
            with open(self.results_cache_file, 'wb') as f:
                pickle.dump(self.transformation_results, f)
            
            # Save success patterns
            with open(self.patterns_cache_file, 'wb') as f:
                pickle.dump(self.success_patterns, f)
            
            # Save model cache
            with open(self.model_cache_file, 'wb') as f:
                pickle.dump(self.model_cache, f)
            
            logger.debug("Caches saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving caches: {e}")
        
        self.cache_stats['save_time'] = (datetime.now() - start_time).total_seconds()
    
    def _generate_code_hash(self, code: str, transformation_type: str = None) -> str:
        """Generate hash for code and transformation type"""
        content = f"{code}_{transformation_type or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_pattern_hash(self, code: str) -> str:
        """Generate hash for code pattern (simplified for pattern matching)"""
        # Simplify code for pattern matching (remove whitespace, normalize)
        simplified = ''.join(code.split()).lower()
        return hashlib.md5(simplified.encode()).hexdigest()
    
    def get_transformation_result(self, code: str, transformation_type: TransformationType) -> Optional[TransformationResult]:
        """Get cached transformation result"""
        self.cache_stats['total_requests'] += 1
        cache_key = self._generate_code_hash(code, transformation_type.value)
        
        if cache_key in self.transformation_results:
            self.cache_stats['hits'] += 1
            return self.transformation_results[cache_key]
        else:
            self.cache_stats['misses'] += 1
            return None
    
    def cache_transformation_result(self, code: str, transformation_type: TransformationType,
                                  output_code: str, success: bool, execution_time: float,
                                  location_context: Dict[str, Any] = None,
                                  metadata: Dict[str, Any] = None):
        """Cache transformation result"""
        cache_key = self._generate_code_hash(code, transformation_type.value)
        
        result = TransformationResult(
            transformation_type=transformation_type.value,
            input_hash=self._generate_code_hash(code),
            output_code=output_code,
            success=success,
            execution_time=execution_time,
            location_context=location_context or {},
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        # Check cache size and evict if necessary
        if len(self.transformation_results) >= self.max_size:
            self._evict_old_entries()
        
        self.transformation_results[cache_key] = result
        
        # Update success patterns
        if success:
            self._update_success_pattern(code, transformation_type, location_context)
    
    def get_success_pattern(self, code: str) -> Optional[SuccessPattern]:
        """Get success pattern for similar code"""
        pattern_hash = self._generate_pattern_hash(code)
        return self.success_patterns.get(pattern_hash)
    
    def _update_success_pattern(self, code: str, transformation_type: TransformationType,
                               location_context: Dict[str, Any] = None):
        """Update success pattern for code"""
        pattern_hash = self._generate_pattern_hash(code)
        
        if pattern_hash not in self.success_patterns:
            self.success_patterns[pattern_hash] = SuccessPattern(
                code_pattern_hash=pattern_hash,
                successful_transformations=[],
                success_rates={},
                location_patterns={},
                last_updated=datetime.now().isoformat(),
                usage_count=0
            )
        
        pattern = self.success_patterns[pattern_hash]
        pattern.usage_count += 1
        pattern.last_updated = datetime.now().isoformat()
        
        # Add transformation to successful list
        if transformation_type.value not in pattern.successful_transformations:
            pattern.successful_transformations.append(transformation_type.value)
            pattern.success_rates[transformation_type.value] = 1.0
        
        # Update location patterns
        if location_context:
            for key, value in location_context.items():
                if key not in pattern.location_patterns:
                    pattern.location_patterns[key] = []
                if str(value) not in pattern.location_patterns[key]:
                    pattern.location_patterns[key].append(str(value))
    
    def get_recommended_transformations(self, code: str, location_context: Dict[str, Any] = None) -> List[TransformationType]:
        """Get recommended transformations based on success patterns"""
        pattern = self.get_success_pattern(code)
        if not pattern:
            return []
        
        # Sort transformations by success rate and usage
        recommended = []
        for transformation_name in pattern.successful_transformations:
            try:
                transformation_type = TransformationType(transformation_name)
                recommended.append(transformation_type)
            except ValueError:
                continue
        
        # Sort by success rate (if available)
        recommended.sort(key=lambda t: pattern.success_rates.get(t.value, 0.0), reverse=True)
        
        return recommended[:10]  # Return top 10 recommendations
    
    def cache_model(self, model_key: str, model_data: Any):
        """Cache model data"""
        self.model_cache[model_key] = {
            'data': model_data,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }
    
    def get_cached_model(self, model_key: str) -> Optional[Any]:
        """Get cached model data"""
        if model_key in self.model_cache:
            self.model_cache[model_key]['access_count'] += 1
            return self.model_cache[model_key]['data']
        return None
    
    def _evict_old_entries(self):
        """Evict old entries from transformation results cache"""
        if not self.transformation_results:
            return
        
        # Sort by timestamp and remove oldest 10%
        entries = list(self.transformation_results.items())
        entries.sort(key=lambda x: x[1].timestamp)
        
        evict_count = max(1, len(entries) // 10)
        for i in range(evict_count):
            key, _ = entries[i]
            del self.transformation_results[key]
            self.cache_stats['evictions'] += 1
        
        logger.debug(f"Evicted {evict_count} old cache entries")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (self.cache_stats['hits'] / self.cache_stats['total_requests'] 
                   if self.cache_stats['total_requests'] > 0 else 0)
        
        return {
            'hit_rate': hit_rate,
            'total_entries': len(self.transformation_results),
            'success_patterns': len(self.success_patterns),
            'cached_models': len(self.model_cache),
            'cache_size_mb': self._estimate_cache_size(),
            **self.cache_stats
        }
    
    def _estimate_cache_size(self) -> float:
        """Estimate cache size in MB"""
        try:
            total_size = 0
            for cache_file in [self.results_cache_file, self.patterns_cache_file, self.model_cache_file]:
                if os.path.exists(cache_file):
                    total_size += os.path.getsize(cache_file)
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def clear_cache(self):
        """Clear all caches"""
        self.transformation_results.clear()
        self.success_patterns.clear()
        self.model_cache.clear()
        
        # Remove cache files
        for cache_file in [self.results_cache_file, self.patterns_cache_file, self.model_cache_file]:
            if os.path.exists(cache_file):
                os.remove(cache_file)
        
        logger.info("All caches cleared")
    
    def cleanup_expired_entries(self, max_age_days: int = 30):
        """Remove expired cache entries"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Clean transformation results
        expired_keys = []
        for key, result in self.transformation_results.items():
            result_date = datetime.fromisoformat(result.timestamp)
            if result_date < cutoff_date:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.transformation_results[key]
        
        # Clean success patterns (less aggressive)
        pattern_cutoff = datetime.now() - timedelta(days=max_age_days * 2)
        expired_patterns = []
        for key, pattern in self.success_patterns.items():
            pattern_date = datetime.fromisoformat(pattern.last_updated)
            if pattern_date < pattern_cutoff and pattern.usage_count < 5:
                expired_patterns.append(key)
        
        for key in expired_patterns:
            del self.success_patterns[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired transformation results and {len(expired_patterns)} expired patterns")
    
    def save_caches(self):
        """Manually save caches"""
        self._save_caches()
    
    def __del__(self):
        """Save caches on destruction"""
        try:
            self._save_caches()
        except:
            pass  # Ignore errors during cleanup
