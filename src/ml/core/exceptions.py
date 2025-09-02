"""
Custom exceptions for the ML betting system.

Provides specific error types for better error handling and debugging.
Following the principle of fail-fast and explicit error communication.
"""

from typing import Optional, Dict, Any


class MLSystemError(Exception):
    """Base exception for all ML system errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class DataProcessingError(MLSystemError):
    """Raised when data processing operations fail."""
    
    def __init__(self, message: str, data_source: Optional[str] = None, 
                 operation: Optional[str] = None, **context):
        context.update({
            'data_source': data_source,
            'operation': operation
        })
        super().__init__(message, context)


class DataValidationError(DataProcessingError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, validation_rule: Optional[str] = None,
                 failed_records: Optional[int] = None, **context):
        context.update({
            'validation_rule': validation_rule,
            'failed_records': failed_records
        })
        super().__init__(message, **context)


class FeatureEngineeringError(MLSystemError):
    """Raised when feature engineering operations fail."""
    
    def __init__(self, message: str, feature_name: Optional[str] = None,
                 feature_type: Optional[str] = None, **context):
        context.update({
            'feature_name': feature_name,
            'feature_type': feature_type
        })
        super().__init__(message, context)


class ModelTrainingError(MLSystemError):
    """Raised when model training fails."""
    
    def __init__(self, message: str, model_type: Optional[str] = None,
                 training_stage: Optional[str] = None, **context):
        context.update({
            'model_type': model_type,
            'training_stage': training_stage
        })
        super().__init__(message, context)


class ModelPredictionError(MLSystemError):
    """Raised when model prediction fails."""
    
    def __init__(self, message: str, model_name: Optional[str] = None,
                 match_id: Optional[str] = None, **context):
        context.update({
            'model_name': model_name,
            'match_id': match_id
        })
        super().__init__(message, context)


class BacktestingError(MLSystemError):
    """Raised when backtesting operations fail."""
    
    def __init__(self, message: str, backtest_period: Optional[str] = None,
                 strategy: Optional[str] = None, **context):
        context.update({
            'backtest_period': backtest_period,
            'strategy': strategy
        })
        super().__init__(message, context)


class ValidationError(MLSystemError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, parameter: Optional[str] = None,
                 expected_type: Optional[str] = None, 
                 actual_value: Optional[Any] = None, **context):
        context.update({
            'parameter': parameter,
            'expected_type': expected_type,
            'actual_value': str(actual_value) if actual_value is not None else None
        })
        super().__init__(message, context)


class CacheError(MLSystemError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, cache_key: Optional[str] = None,
                 operation: Optional[str] = None, **context):
        context.update({
            'cache_key': cache_key,
            'operation': operation
        })
        super().__init__(message, context)


class ConfigurationError(MLSystemError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_file: Optional[str] = None, **context):
        context.update({
            'config_key': config_key,
            'config_file': config_file
        })
        super().__init__(message, context)


class ExternalServiceError(MLSystemError):
    """Raised when external service calls fail."""
    
    def __init__(self, message: str, service_name: Optional[str] = None,
                 endpoint: Optional[str] = None, status_code: Optional[int] = None,
                 **context):
        context.update({
            'service_name': service_name,
            'endpoint': endpoint,
            'status_code': status_code
        })
        super().__init__(message, context)