"""Shared constants for the AI Job Scraper application.

This module defines constants used across scraping modules, such as regex patterns
for filtering AI-related job titles and default search keywords/locations.
"""

import re

RELEVANT_PHRASES = [
    "ai",
    "artificial intelligence",
    "ml",
    "machine learning",
    "data science",
    "data scientist",
    "data engineer",
    "nlp",
    "natural language processing",
    "computer vision",
    "deep learning",
    "ai engineer",
    "ai agent engineer",
    "ai agent",
    "agentic ai engineer",
    "ai researcher",
    "research engineer",
    "mlops",
    "machine learning engineer",
    "ml engineer",
    "senior ml engineer",
    "staff ml engineer",
    "principal ml engineer",
    "ai software engineer",
    "ml infrastructure engineer",
    "mlops engineer",
    "deep learning engineer",
    "computer vision engineer",
    "nlp engineer",
    "speech recognition engineer",
    "reinforcement learning engineer",
    "ai research scientist",
    "machine learning researcher",
    "research scientist",
    "applied scientist",
    "principal researcher",
    "generative ai engineer",
    "rag engineer",
    "retrieval-augmented generation developer",
    "rag pipeline engineer",
    "ai agent developer",
    "gpu machine learning engineer",
    "cuda engineer",
    "performance engineer",
    "deep learning compiler engineer",
    "gpgpu engineer",
    "ml acceleration engineer",
    "ai hardware engineer",
    "cuda libraries engineer",
    "tensorrt engineer",
    "ai solutions architect",
    "ai architect",
    "ai platform architect",
    "agentic",
]
AI_REGEX = re.compile(
    r"(?i)\b(" + "|".join(re.escape(p) for p in RELEVANT_PHRASES) + r")\b",
)

SEARCH_KEYWORDS = ["ai", "machine learning", "data science"]
SEARCH_LOCATIONS = ["USA", "Remote"]

# Salary filtering constants
SALARY_UNBOUNDED_THRESHOLD = 750000  # Values at or above this are treated as unbounded
SALARY_DEFAULT_MIN = 0
SALARY_DEFAULT_MAX = SALARY_UNBOUNDED_THRESHOLD

# UI constants for salary slider
SALARY_SLIDER_STEP = 25000
SALARY_SLIDER_FORMAT = "$%dk"

# Application status options
APPLICATION_STATUSES = ["New", "Interested", "Applied", "Rejected"]
