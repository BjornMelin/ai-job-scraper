"""Task Complexity Analyzer for intelligent AI routing decisions.

This module analyzes task complexity to determine the optimal AI service
(local vs cloud) based on various factors like content complexity, token count,
task type, and performance requirements.
"""

from __future__ import annotations

import logging
import re

from typing import Any, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ComplexityAnalysis(BaseModel):
    """Results of task complexity analysis."""

    complexity_score: float = Field(
        description="Overall complexity score (0.0-1.0, higher = more complex)",
        ge=0.0,
        le=1.0,
    )
    token_count: int = Field(description="Estimated token count for the task")
    content_complexity: float = Field(
        description="Content complexity score (0.0-1.0)", ge=0.0, le=1.0
    )
    task_type_complexity: float = Field(
        description="Task type complexity score (0.0-1.0)", ge=0.0, le=1.0
    )
    reasoning_requirement: float = Field(
        description="Reasoning requirement score (0.0-1.0)", ge=0.0, le=1.0
    )
    structural_complexity: float = Field(
        description="Output structure complexity score (0.0-1.0)", ge=0.0, le=1.0
    )
    recommended_service: str = Field(
        description="Recommended AI service: 'local', 'cloud', or 'either'"
    )
    confidence: float = Field(
        description="Confidence in the recommendation (0.0-1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Explanation of the complexity assessment")


class TaskComplexityAnalyzer:
    """Analyzes task complexity to enable intelligent AI routing decisions.

    Features:
    - Content complexity analysis based on technical terms and concepts
    - Task type classification and complexity scoring
    - Token counting and context size evaluation
    - Reasoning requirement assessment
    - Output structure complexity analysis
    - Confidence scoring for routing decisions
    """

    def __init__(self) -> None:
        """Initialize the Task Complexity Analyzer."""
        # Technical terms that indicate higher complexity
        self.technical_terms = {
            "programming": [
                "algorithm",
                "function",
                "class",
                "method",
                "variable",
                "loop",
                "condition",
                "recursive",
                "async",
                "concurrency",
                "database",
                "API",
                "framework",
            ],
            "finance": [
                "revenue",
                "profit",
                "loss",
                "equity",
                "liability",
                "depreciation",
                "amortization",
                "valuation",
                "investment",
                "portfolio",
                "risk",
                "volatility",
            ],
            "legal": [
                "contract",
                "liability",
                "compliance",
                "regulation",
                "statute",
                "precedent",
                "jurisdiction",
                "litigation",
                "arbitration",
                "intellectual property",
            ],
            "medical": [
                "diagnosis",
                "treatment",
                "symptoms",
                "medication",
                "pathology",
                "anatomy",
                "physiology",
                "clinical",
                "therapeutic",
                "pharmaceutical",
            ],
            "scientific": [
                "hypothesis",
                "methodology",
                "analysis",
                "statistical",
                "correlation",
                "experimental",
                "theoretical",
                "empirical",
                "quantitative",
                "qualitative",
            ],
        }

        # Task type patterns and their complexity scores
        self.task_patterns = {
            "simple_extraction": {
                "patterns": [r"extract\s+(\w+)", r"find\s+the\s+(\w+)", r"get\s+(\w+)"],
                "complexity": 0.2,
            },
            "structured_extraction": {
                "patterns": [
                    r"extract.*into.*format",
                    r"parse.*structure",
                    r"convert.*to.*json",
                ],
                "complexity": 0.4,
            },
            "analysis": {
                "patterns": [
                    r"analyze",
                    r"evaluate",
                    r"assess",
                    r"compare",
                    r"contrast",
                ],
                "complexity": 0.6,
            },
            "reasoning": {
                "patterns": [
                    r"explain\s+why",
                    r"reason",
                    r"justify",
                    r"deduce",
                    r"infer",
                ],
                "complexity": 0.8,
            },
            "creative": {
                "patterns": [
                    r"generate",
                    r"create",
                    r"write.*story",
                    r"compose",
                    r"design",
                ],
                "complexity": 0.7,
            },
            "complex_reasoning": {
                "patterns": [
                    r"solve.*problem",
                    r"debug",
                    r"troubleshoot",
                    r"optimize",
                    r"prove",
                ],
                "complexity": 0.9,
            },
        }

        # Reasoning indicators
        self.reasoning_indicators = [
            "why",
            "how",
            "explain",
            "because",
            "therefore",
            "consequently",
            "reason",
            "cause",
            "effect",
            "analyze",
            "compare",
            "contrast",
            "evaluate",
            "assess",
            "judge",
            "decide",
            "choose",
            "recommend",
        ]

        logger.info("Task Complexity Analyzer initialized")

    def analyze_content_complexity(self, text: str) -> float:
        """Analyze the complexity of content based on technical terms and concepts.

        Args:
            text: Text content to analyze

        Returns:
            Complexity score between 0.0 and 1.0
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        total_score = 0.0
        term_count = 0

        # Check for technical terms across domains
        for domain, terms in self.technical_terms.items():
            domain_matches = 0
            for term in terms:
                if term in text_lower:
                    domain_matches += 1
                    term_count += 1

            # Higher weight for domains with more matches (specialization)
            if domain_matches > 0:
                domain_score = min(domain_matches / len(terms), 1.0)
                total_score += domain_score * (1.0 + domain_matches * 0.1)

        # Normalize score
        if term_count == 0:
            content_complexity = 0.1  # Base complexity for any text
        else:
            content_complexity = min(total_score / len(self.technical_terms), 1.0)

        # Additional complexity indicators
        sentence_count = len(re.findall(r"[.!?]+", text))
        avg_sentence_length = len(text.split()) / max(sentence_count, 1)

        # Longer sentences and more complex punctuation indicate higher complexity
        if avg_sentence_length > 20:
            content_complexity += 0.1
        if avg_sentence_length > 30:
            content_complexity += 0.1

        # Complex punctuation patterns
        if re.search(r"[;:()—–-]", text):
            content_complexity += 0.1

        return min(content_complexity, 1.0)

    def analyze_task_type_complexity(self, text: str) -> tuple[float, str]:
        """Analyze task type and determine its complexity.

        Args:
            text: Task description text

        Returns:
            Tuple of (complexity_score, task_type)
        """
        if not text:
            return 0.2, "unknown"

        text_lower = text.lower()
        best_match_complexity = 0.2
        best_match_type = "simple_extraction"

        for task_type, config in self.task_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    if config["complexity"] > best_match_complexity:
                        best_match_complexity = config["complexity"]
                        best_match_type = task_type
                        break

        return best_match_complexity, best_match_type

    def analyze_reasoning_requirement(self, text: str) -> float:
        """Analyze how much reasoning the task requires.

        Args:
            text: Task description text

        Returns:
            Reasoning requirement score between 0.0 and 1.0
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        reasoning_score = 0.0

        # Count reasoning indicators
        for indicator in self.reasoning_indicators:
            if indicator in text_lower:
                reasoning_score += 0.1

        # Look for complex reasoning patterns
        complex_patterns = [
            r"step\s+by\s+step",
            r"break\s+down",
            r"walkthrough",
            r"explain.*process",
            r"analyze.*relationship",
            r"compare.*and.*contrast",
            r"pros.*and.*cons",
            r"advantages.*disadvantages",
        ]

        for pattern in complex_patterns:
            if re.search(pattern, text_lower):
                reasoning_score += 0.2

        return min(reasoning_score, 1.0)

    def analyze_structural_complexity(
        self, response_model: Any = None, **kwargs: Any
    ) -> float:
        """Analyze the complexity of the expected output structure.

        Args:
            response_model: Pydantic model for structured output (if any)
            **kwargs: Additional parameters that might indicate structure complexity

        Returns:
            Structural complexity score between 0.0 and 1.0
        """
        complexity = 0.0

        if response_model is not None:
            # Analyze Pydantic model complexity
            try:
                # Check if it's a Pydantic model
                if hasattr(response_model, "__fields__"):
                    fields = response_model.__fields__
                    field_count = len(fields)

                    # Base complexity from field count
                    complexity += min(field_count * 0.1, 0.5)

                    # Additional complexity for nested models
                    for field_info in fields.values():
                        field_type = field_info.type_

                        # Check for nested models
                        if hasattr(field_type, "__fields__"):
                            complexity += 0.2

                        # Check for complex types (lists, unions, etc.)
                        if hasattr(field_type, "__origin__"):
                            if field_type.__origin__ in (list, tuple, set):
                                complexity += 0.15
                            elif field_type.__origin__ is Union:  # type: ignore
                                complexity += 0.1

            except Exception:
                # If we can't analyze the model, assume medium complexity
                complexity = 0.4

        # Check for JSON formatting requirements
        if kwargs.get("format") == "json" or kwargs.get("response_format") == "json":
            complexity += 0.2

        # Check for specific output formatting requirements
        format_indicators = kwargs.get("extra_body", {})
        if format_indicators:
            complexity += 0.1

        return min(complexity, 1.0)

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation).

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Rough estimation: ~0.75 tokens per word for English text
        word_count = len(text.split())
        return int(word_count * 0.75)

    def analyze_task_complexity(
        self,
        messages: list[dict[str, str]] | None = None,
        prompt: str | None = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> ComplexityAnalysis:
        """Perform comprehensive task complexity analysis.

        Args:
            messages: Chat messages (if using chat format)
            prompt: Direct prompt text (if using completion format)
            response_model: Expected response model for structured output
            **kwargs: Additional parameters affecting complexity

        Returns:
            ComplexityAnalysis with detailed scoring and recommendation
        """
        # Combine all text for analysis
        text_content = ""
        if messages:
            text_content = " ".join(msg.get("content", "") for msg in messages)
        elif prompt:
            text_content = prompt

        if not text_content:
            # No content to analyze - assume simple task
            return ComplexityAnalysis(
                complexity_score=0.1,
                token_count=0,
                content_complexity=0.0,
                task_type_complexity=0.1,
                reasoning_requirement=0.0,
                structural_complexity=0.0,
                recommended_service="local",
                confidence=0.8,
                reasoning="No content provided - defaulting to simple local processing",
            )

        # Perform individual analyses
        content_complexity = self.analyze_content_complexity(text_content)
        task_type_complexity, task_type = self.analyze_task_type_complexity(
            text_content
        )
        reasoning_requirement = self.analyze_reasoning_requirement(text_content)
        structural_complexity = self.analyze_structural_complexity(
            response_model, **kwargs
        )
        token_count = self.estimate_token_count(text_content)

        # Calculate overall complexity score (weighted average)
        complexity_score = (
            content_complexity * 0.25
            + task_type_complexity * 0.3
            + reasoning_requirement * 0.25
            + structural_complexity * 0.2
        )

        # Adjust for token count (large contexts need cloud)
        if token_count > 8000:
            complexity_score = max(complexity_score, 0.6)
        elif token_count > 4000:
            complexity_score = max(complexity_score, 0.4)

        # Make routing recommendation
        if complexity_score < 0.3 and token_count < 4000:
            recommended_service = "local"
            confidence = 0.9
            reasoning = f"Low complexity task ({complexity_score:.2f}) suitable for local processing"
        elif complexity_score > 0.7 or token_count > 8000:
            recommended_service = "cloud"
            confidence = 0.85
            reasoning = f"High complexity task ({complexity_score:.2f}) requiring cloud capabilities"
        else:
            recommended_service = "either"
            confidence = 0.6
            reasoning = f"Medium complexity task ({complexity_score:.2f}) - either service could work"

        # Adjust confidence based on analysis certainty
        if content_complexity > 0.5 and reasoning_requirement > 0.5:
            confidence += 0.1  # High confidence in complex tasks
        elif content_complexity < 0.2 and reasoning_requirement < 0.2:
            confidence += 0.1  # High confidence in simple tasks

        confidence = min(confidence, 1.0)

        return ComplexityAnalysis(
            complexity_score=complexity_score,
            token_count=token_count,
            content_complexity=content_complexity,
            task_type_complexity=task_type_complexity,
            reasoning_requirement=reasoning_requirement,
            structural_complexity=structural_complexity,
            recommended_service=recommended_service,
            confidence=confidence,
            reasoning=reasoning + f" (task type: {task_type}, tokens: {token_count})",
        )

    def should_use_cloud(
        self,
        complexity_analysis: ComplexityAnalysis,
        local_service_available: bool = True,
        cloud_service_available: bool = True,
        cost_preference: str = "balanced",  # "cost_first", "balanced", "quality_first"
    ) -> tuple[bool, str]:
        """Determine whether to use cloud service based on analysis and preferences.

        Args:
            complexity_analysis: Results from analyze_task_complexity
            local_service_available: Whether local service is healthy/available
            cloud_service_available: Whether cloud service is healthy/available
            cost_preference: Preference for cost vs quality tradeoffs

        Returns:
            Tuple of (use_cloud: bool, reasoning: str)
        """
        # Check availability first
        if not local_service_available and cloud_service_available:
            return True, "Local service unavailable, using cloud fallback"
        if not cloud_service_available and local_service_available:
            return False, "Cloud service unavailable, using local service"
        if not local_service_available and not cloud_service_available:
            return False, "No services available - will attempt local as last resort"

        # Both services available - make intelligent choice
        complexity = complexity_analysis.complexity_score
        confidence = complexity_analysis.confidence

        # Apply cost preference
        if cost_preference == "cost_first":
            # Bias toward local (free) service
            threshold = 0.6
        elif cost_preference == "quality_first":
            # Bias toward cloud (higher capability) service
            threshold = 0.3
        else:  # balanced
            threshold = 0.5

        # Make decision based on complexity and confidence
        if complexity > threshold:
            if confidence > 0.7:
                return (
                    True,
                    f"High complexity ({complexity:.2f}) with high confidence ({confidence:.2f}) - using cloud",
                )
            return (
                True,
                f"High complexity ({complexity:.2f}) but low confidence - using cloud for safety",
            )
        if confidence > 0.8:
            return (
                False,
                f"Low complexity ({complexity:.2f}) with high confidence ({confidence:.2f}) - using local",
            )
        # Medium confidence - consider token count
        if complexity_analysis.token_count > 6000:
            return True, "Medium confidence but large context - using cloud"
        return (
            False,
            "Medium confidence and manageable context - using local",
        )


# Module-level singleton for easy access
_complexity_analyzer: TaskComplexityAnalyzer | None = None


def get_complexity_analyzer() -> TaskComplexityAnalyzer:
    """Get singleton instance of TaskComplexityAnalyzer.

    Returns:
        TaskComplexityAnalyzer singleton instance
    """
    global _complexity_analyzer
    if _complexity_analyzer is None:
        _complexity_analyzer = TaskComplexityAnalyzer()
    return _complexity_analyzer


def reset_complexity_analyzer() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _complexity_analyzer
    _complexity_analyzer = None
