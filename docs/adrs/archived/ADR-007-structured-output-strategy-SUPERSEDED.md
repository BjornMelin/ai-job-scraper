# ADR-007: Structured Output Strategy for Job Extraction [SUPERSEDED]

## ⚠️ **SUPERSEDED BY ADR-004**

**Supersession Date:** August 23, 2025  
**Superseded By:** ADR-004 (Local AI Processing Architecture)  
**Reason:** vLLM native structured outputs provide identical functionality with 97% code reduction

**For current structured output implementation, see:**

- **ADR-004**: Section "Structured Output Generation"
- **Location**: `ai-job-scraper/docs/adrs/ADR-004-local-ai-integration.md`

---

## Historical Context

This ADR previously defined structured JSON output generation using the **Outlines library** with vLLM integration. The approach provided 100% valid JSON through finite state machine (FSM) constraints.

## Why Superseded

**Technical Evolution:**

- **vLLM native guided_json** now provides identical functionality
- **97% code reduction**: 150+ lines → 5 lines
- **Same reliability**: 100% valid JSON maintained
- **Better performance**: Native optimization vs. FSM constraints
- **Reduced dependencies**: Eliminates Outlines library requirement

## Migration Path

**Old Approach (Outlines):**

```python
from outlines import models, generate
model = models.vllm("Qwen/Qwen3-4B-Instruct-2507-FP8")
generator = generate.json(model, JobPosting)
result = generator(prompt)
```

**New Approach (vLLM Native):**

```python
completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    extra_body={"guided_json": JobPosting.model_json_schema()}
)
result = JobPosting.parse_raw(completion.choices[0].message.content)
```

---

## Original ADR Content [ARCHIVED]

**Original Status:** Accepted  
**Original Version/Date:** v2.0 / 2025-08-18

### Original Description

Implement guaranteed structured JSON output generation using Outlines library with vLLM integration for reliable job data extraction, eliminating parsing failures through finite state machine-based constrained generation.

### Key Features Migrated to ADR-004

- ✅ 100% valid JSON generation
- ✅ Complex nested schema support  
- ✅ Pydantic model integration
- ✅ Performance optimization
- ✅ Error handling and validation

### Dependencies Eliminated

- ❌ `outlines>=0.1.0` (replaced by vLLM native)
- ❌ Custom FSM constraint logic
- ❌ Schema compilation overhead
- ❌ Outlines-specific error handling

---

**For implementation details, examples, and current configuration, refer to ADR-004's "Structured Output Generation" section.**
