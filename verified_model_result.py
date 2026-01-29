#!/usr/bin/env python3
"""
Verified Model Result Dataclasses

Dataclasses for tracking verified checker execution and evaluation results.
These provide explicit tracking of whether results are verified vs assumed.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class VerifiedCheckerResult:
    """
    Result of a verified checker execution.
    
    This explicitly tracks whether the checker ran successfully,
    crashed, or had other issues that make the result unreliable.
    """
    # Core success/failure
    checker_ran: bool  # True if checker executed without fatal crash
    crashed: bool  # True if a crash was detected
    
    # Crash details
    crash_reason: Optional[str] = None
    crash_indicators_found: List[str] = field(default_factory=list)
    has_stack_trace: bool = False
    
    # Compilation status (separate from crash)
    compilation_success: bool = True  # True if code compiled (may still have errors)
    has_compilation_errors: bool = False  # True if compilation errors were present
    
    # Processing verification
    files_processed: bool = False  # True if checker appears to have processed files
    files_count: int = 0  # Number of files mentioned in output
    
    # Warning parsing
    warnings_verified: bool = False  # True if warnings were successfully parsed
    warning_count: int = 0  # Parsed warning count
    error_count: int = 0  # Parsed error count (compilation errors)
    
    # Raw data for debugging
    raw_output: str = ""  # Full checker output
    returncode: int = 0  # Process return code
    
    # Confidence in this result
    confidence: float = 0.0  # 0.0 to 1.0
    
    # Timing
    execution_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def is_reliable(self) -> bool:
        """Check if this result is reliable enough to use"""
        return (
            self.checker_ran and 
            not self.crashed and 
            self.confidence >= 0.7
        )
    
    def get_warning_count(self) -> Optional[int]:
        """
        Get warning count, or None if unreliable.
        
        Returns:
            Warning count if verified, None if result is unreliable
        """
        if self.is_reliable() and self.warnings_verified:
            return self.warning_count
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def __str__(self) -> str:
        status = "CRASHED" if self.crashed else ("OK" if self.checker_ran else "FAILED")
        return (
            f"VerifiedCheckerResult({status}, "
            f"warnings={self.warning_count}, "
            f"confidence={self.confidence:.2f})"
        )


@dataclass
class VerifiedModelEvaluationResult:
    """
    Result of a verified model evaluation.
    
    Extends the basic evaluation result with verification status
    for both baseline and post-placement checker runs.
    """
    # Model identification
    base_model: str
    
    # Verification status
    verified: bool = False  # True if both baseline and post-placement are verified
    verification_error: Optional[str] = None  # Error message if not verified
    
    # Baseline checker verification
    baseline_verified: bool = False
    baseline_checker_result: Optional[VerifiedCheckerResult] = None
    
    # Post-placement checker verification
    post_placement_verified: bool = False
    post_placement_checker_result: Optional[VerifiedCheckerResult] = None
    
    # Annotation placement
    annotations_placed: int = 0
    placement_success: bool = False
    
    # Warning counts (None if not verified)
    baseline_warnings: Optional[int] = None
    warnings_after: Optional[int] = None
    
    # Calculated metrics (None if not verified)
    warning_reduction: Optional[int] = None
    reduction_percentage: Optional[float] = None
    
    # Legacy compatibility fields
    compilation_success: bool = True
    error_message: Optional[str] = None
    
    # Saved outputs for debugging
    checker_output_saved: Optional[str] = None  # Path to saved output file
    
    # Timing
    evaluation_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def calculate_reduction(self) -> None:
        """Calculate warning reduction if both counts are verified"""
        if self.baseline_warnings is not None and self.warnings_after is not None:
            self.warning_reduction = self.baseline_warnings - self.warnings_after
            if self.baseline_warnings > 0:
                self.reduction_percentage = (self.warning_reduction / self.baseline_warnings) * 100.0
            else:
                self.reduction_percentage = 0.0
        else:
            self.warning_reduction = None
            self.reduction_percentage = None
    
    def get_confidence(self) -> float:
        """Get overall confidence in this result"""
        if not self.verified:
            return 0.0
        
        confidences = []
        if self.baseline_checker_result:
            confidences.append(self.baseline_checker_result.confidence)
        if self.post_placement_checker_result:
            confidences.append(self.post_placement_checker_result.confidence)
        
        if confidences:
            return min(confidences)  # Use minimum confidence
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'base_model': self.base_model,
            'verified': self.verified,
            'verification_error': self.verification_error,
            'baseline_verified': self.baseline_verified,
            'post_placement_verified': self.post_placement_verified,
            'annotations_placed': self.annotations_placed,
            'placement_success': self.placement_success,
            'baseline_warnings': self.baseline_warnings,
            'warnings_after': self.warnings_after,
            'warning_reduction': self.warning_reduction,
            'reduction_percentage': self.reduction_percentage,
            'compilation_success': self.compilation_success,
            'error_message': self.error_message,
            'checker_output_saved': self.checker_output_saved,
            'evaluation_time_seconds': self.evaluation_time_seconds,
            'timestamp': self.timestamp,
            'confidence': self.get_confidence(),
        }
        
        # Include checker results if available
        if self.baseline_checker_result:
            result['baseline_checker_result'] = self.baseline_checker_result.to_dict()
        if self.post_placement_checker_result:
            result['post_placement_checker_result'] = self.post_placement_checker_result.to_dict()
        
        return result
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy format compatible with existing evaluation report.
        
        This allows the verified results to be used with existing tooling.
        """
        return {
            'base_model': self.base_model,
            'annotations_placed': self.annotations_placed,
            'warnings_after': self.warnings_after if self.warnings_after is not None else 0,
            'warning_reduction': self.warning_reduction if self.warning_reduction is not None else 0,
            'reduction_percentage': self.reduction_percentage if self.reduction_percentage is not None else 0.0,
            'placement_success': self.placement_success,
            'compilation_success': self.compilation_success,
            'error_message': self.error_message,
            # Additional verified fields
            'verified': self.verified,
            'verification_error': self.verification_error,
            'confidence': self.get_confidence(),
        }
    
    def __str__(self) -> str:
        verified_str = "VERIFIED" if self.verified else "UNVERIFIED"
        if self.warning_reduction is not None:
            return (
                f"VerifiedModelEvaluationResult({self.base_model}, {verified_str}, "
                f"reduction={self.warning_reduction} ({self.reduction_percentage:.1f}%))"
            )
        else:
            return (
                f"VerifiedModelEvaluationResult({self.base_model}, {verified_str}, "
                f"error={self.verification_error})"
            )


@dataclass
class VerifiedProjectEvaluationResult:
    """
    Result of a verified project evaluation.
    
    Contains verified results for all models tested on a project.
    """
    project_name: str
    project_url: str
    
    # Baseline verification
    baseline_warnings: Optional[int] = None
    baseline_verified: bool = False
    baseline_verification_error: Optional[str] = None
    
    # Model results
    model_results: List[VerifiedModelEvaluationResult] = field(default_factory=list)
    
    # Overall project status
    error_message: Optional[str] = None
    all_verified: bool = False
    
    # Timing
    evaluation_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def calculate_verification_status(self) -> None:
        """Update all_verified based on model results"""
        if not self.model_results:
            self.all_verified = False
            return
        
        self.all_verified = all(r.verified for r in self.model_results)
    
    def get_verified_count(self) -> int:
        """Get count of verified model results"""
        return sum(1 for r in self.model_results if r.verified)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'project_name': self.project_name,
            'project_url': self.project_url,
            'baseline_warnings': self.baseline_warnings,
            'baseline_verified': self.baseline_verified,
            'baseline_verification_error': self.baseline_verification_error,
            'model_results': [r.to_dict() for r in self.model_results],
            'error_message': self.error_message,
            'all_verified': self.all_verified,
            'verified_count': self.get_verified_count(),
            'total_count': len(self.model_results),
            'evaluation_time_seconds': self.evaluation_time_seconds,
            'timestamp': self.timestamp,
        }
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy format for compatibility"""
        return {
            'project_name': self.project_name,
            'project_url': self.project_url,
            'baseline_warnings': self.baseline_warnings if self.baseline_warnings is not None else 0,
            'model_results': [r.to_legacy_dict() for r in self.model_results],
            'error_message': self.error_message,
        }


@dataclass
class VerifiedEvaluationReport:
    """
    Complete verified evaluation report.
    """
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: List[VerifiedProjectEvaluationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        # Calculate summary statistics
        total_models = sum(len(r.model_results) for r in self.results)
        verified_models = sum(r.get_verified_count() for r in self.results)
        
        return {
            'metadata': {
                **self.metadata,
                'verified': True,
                'total_model_results': total_models,
                'verified_model_results': verified_models,
                'verification_rate': verified_models / total_models if total_models > 0 else 0,
            },
            'results': [r.to_dict() for r in self.results],
        }
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy format for compatibility"""
        return {
            'metadata': self.metadata,
            'results': [r.to_legacy_dict() for r in self.results],
        }
