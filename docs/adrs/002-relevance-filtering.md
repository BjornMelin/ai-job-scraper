# ADR-002: Relevance Filtering

## Title

Mechanism for Filtering Relevant Job Postings

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

Post-scraping, jobs must be filtered for relevance to AI/ML roles (e.g., "AI Engineer", "MLOps") to avoid noise. Initial keyword-based for speed, with modularity for future semantic upgrades.

## Related Requirements

- Regex matching for titles like "(AI|Machine Learning|MLOps|AI Agent).*Engineer".
- Applied before DB storage to optimize.

## Alternatives

- LLM-based filtering (e.g., Ollama classify): More accurate but slower and requires setup.
- Simple string contains: Misses variations and case sensitivity.

## Decision

Implement regex filtering with re.compile(r"(AI|Machine Learning|MLOps|AI Agent).*Engineer", re.I) for efficiency and case-insensitivity. Design modular (e.g., is_relevant func) for easy swap to LLM if needed.

## Related Decisions

- ADR-001 (Filtering occurs after extraction).
- ADR-003 (Only relevant jobs stored/updated).

## Design

- **Filter Function**: def is_relevant(job): return RELEVANT_KEYWORDS.search(job["title"]).
- **Integration**: After scraping, relevant = [j for j in jobs if is_relevant(j)]; Pass to update_db.
- **Implementation Notes**: Case-insensitive (re.I). Future toggle: if AppSettings.use_llm_filter: Use local Ollama to classify semantic relevance.
- **Testing**: def test_is_relevant(): assert is_relevant({"title": "Senior AI Engineer"}); assert not is_relevant({"title": "Marketing Manager"}); Test edge cases like "Machine Learning Ops Engineer".

## Consequences

- Fast and lightweight filtering (regex suitable for MVP performance).
- Accurate for keyword matches, modular for enhancements (e.g., LLM integration later).

- Potential false negatives/positives (mitigated by future semantic upgrade if user feedback indicates need).

**Changelog:**  

- 1.0 (July 29, 2025): Defined initial regex approach with modularity for LLM.
