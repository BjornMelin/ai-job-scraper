"""Hybrid AI Integration Module for AI Job Scraper.

This module provides a comprehensive hybrid AI architecture that combines:
- Local vLLM inference for fast, cost-effective processing
- Cloud AI fallback for complex tasks requiring advanced capabilities
- Intelligent routing based on task complexity assessment
- Structured output processing with Instructor for enhanced reliability

The hybrid approach optimizes for both performance and cost while ensuring
graceful degradation when local resources are unavailable.
"""

from .background_ai_processor import BackgroundAIProcessor, get_background_ai_processor
from .cloud_ai_service import CloudAIService, get_cloud_ai_service
from .hybrid_ai_router import HybridAIRouter, get_hybrid_ai_router
from .local_vllm_service import LocalVLLMService, get_local_vllm_service
from .structured_output_processor import (
    StructuredOutputProcessor,
    get_structured_output_processor,
)
from .task_complexity_analyzer import TaskComplexityAnalyzer, get_complexity_analyzer

__all__ = [
    "BackgroundAIProcessor",
    "CloudAIService",
    "HybridAIRouter",
    "LocalVLLMService",
    "StructuredOutputProcessor",
    "TaskComplexityAnalyzer",
    # Singleton accessor functions
    "get_background_ai_processor",
    "get_cloud_ai_service",
    "get_hybrid_ai_router",
    "get_local_vllm_service",
    "get_structured_output_processor",
    "get_complexity_analyzer",
]
