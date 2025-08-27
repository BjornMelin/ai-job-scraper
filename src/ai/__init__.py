"""AI Integration Module for AI Job Scraper.

This module provides essential AI services for the job scraper application:
- Centralized AI client (ai_client.py) for all AI operations
- Local AI processor with Instructor integration for structured outputs
- Local vLLM service for AI inference when available

The architecture has been simplified to eliminate complexity and focus on
core functionality with library-first implementations.
"""

# Import centralized AI client from root src
from src.ai_client import get_ai_client

from .local_processor import (
    JobExtraction,
    LocalAIProcessor,
    enhance_job_description,
    extract_job_skills,
)
from .local_vllm_service import LocalVLLMService, local_service

__all__ = [
    "JobExtraction",
    "LocalAIProcessor",
    "LocalVLLMService",
    "enhance_job_description",
    "extract_job_skills",
    "get_ai_client",
    "local_service",
]
