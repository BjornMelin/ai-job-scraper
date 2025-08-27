"""Local AI processor using LiteLLM + Instructor for structured output."""

from __future__ import annotations

import instructor

from litellm import completion
from pydantic import BaseModel, Field

from src.ai_client import get_ai_client


class JobExtraction(BaseModel):
    """Structured job extraction schema."""

    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str | None = Field(None, description="Job location")
    description: str = Field(..., description="Job description")
    requirements: list[str] = Field(
        default_factory=list, description="Job requirements"
    )
    benefits: list[str] = Field(default_factory=list, description="Job benefits")


class LocalAIProcessor:
    """AI processor using LiteLLM + Instructor for structured outputs."""

    def __init__(self) -> None:
        """Initialize with Instructor-wrapped LiteLLM client."""
        self.client = instructor.from_litellm(completion=completion)

    async def extract_jobs(self, content: str) -> JobExtraction:
        """Extract structured job data using Instructor.

        Args:
            content: Raw job posting content to extract from

        Returns:
            JobExtraction: Structured job data
        """
        return await self.client.chat.completions.create(
            model="local-qwen",  # Routes via LiteLLM config
            response_model=JobExtraction,
            messages=[
                {
                    "role": "system",
                    "content": "Extract job information from the content.",
                },
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=2000,
        )


# Backward compatibility functions
async def enhance_job_description(job_data: dict) -> str:
    """Enhance job description using AI."""
    client = get_ai_client()
    description = job_data.get("description", "")
    messages = [
        {"role": "system", "content": "Enhance and improve the job description."},
        {
            "role": "user",
            "content": f"Improve this job description: {description}",
        },
    ]
    return client.get_simple_completion(messages, model="local-qwen")


async def extract_job_skills(job_data: dict) -> list[str]:
    """Extract skills from job posting."""
    processor = LocalAIProcessor()
    result = await processor.extract_jobs(job_data.get("description", ""))
    return result.requirements
