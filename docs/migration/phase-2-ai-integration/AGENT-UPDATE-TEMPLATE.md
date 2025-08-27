# Agent Update Template for MIGRATION-LOG.md

## Copy-Paste Template for Status Updates

```markdown
**[YYYY-MM-DDTHH:MM:SSZ] - [AGENT_NAME]**: ✅ [TASK] completed
- Evidence: [specific proof - file paths, line numbers, test results]
- Library used: [which proven library and specific features]
- Code reduction: [lines eliminated/added with before/after counts]
- Complexity justification: [KISS/DRY/YAGNI adherence explanation]
```

## Examples

### Successful Task Completion

```markdown
**2025-08-27T21:45:00Z - AI-INTEGRATION-AGENT**: ✅ LiteLLM configuration validated
- Evidence: /config/litellm.yaml updated, health check passing at http://localhost:4000/health
- Library used: LiteLLM proxy server with OpenAI-compatible routing
- Code reduction: 1,200 lines eliminated (custom AI client → 15 lines config)
- Complexity justification: KISS - configuration over code, proven library patterns
```

### Checkpoint Update

```markdown
**2025-08-27T22:15:00Z - AI-CLIENT-AGENT**: ✅ AI client created (15 lines)
- Evidence: /src/ai_client.py - 15 lines, passes test_ai_client_integration.py
- Library used: OpenAI SDK with LiteLLM proxy endpoint
- Code reduction: 800+ lines removed from /src/ai/*, replaced with 15 lines
- Complexity justification: DRY - reusing OpenAI patterns, no reinvention
```

### Issue Encountered

```markdown
**2025-08-27T22:30:00Z - ENVIRONMENT-AGENT**: ⚠️ Ollama cleanup blocked - dependency references found
- Evidence: rg "ollama" found 12 references in /tests/ and /config/
- Library used: ripgrep for dependency analysis
- Code reduction: Pending - cleanup required before proceeding
- Complexity justification: Safety-first - verify all references before deletion
```

## Required Fields

1. **Timestamp**: ISO format (YYYY-MM-DDTHH:MM:SSZ)
2. **Agent Name**: Specific role (e.g., AI-INTEGRATION-AGENT, ENVIRONMENT-AGENT)
3. **Status**: ✅ (complete), 🔄 (in progress), ⚠️ (blocked), ❌ (failed)
4. **Evidence**: Concrete proof (file paths, test results, URLs)
5. **Library**: Specific library/feature used
6. **Code Reduction**: Before/after line counts
7. **Complexity**: KISS/DRY/YAGNI adherence explanation

## Quality Standards

- **Specific over vague**: "/src/ai_client.py:15" not "updated client"
- **Measurable progress**: "removed 800 lines" not "simplified code"  
- **Library justification**: "using LiteLLM proxy" not "improved AI"
- **Evidence-based**: Include test results, file paths, health checks
