"""AI Integration Module for AI Job Scraper.

This module provides essential AI services for the job scraper application:
- Local vLLM service for AI inference when available

The architecture has been simplified to eliminate complexity and focus on
core functionality with library-first implementations.
"""

from .local_vllm_service import LocalVLLMService, get_local_vllm_service

__all__ = [
    "LocalVLLMService",
    "get_local_vllm_service",
]
