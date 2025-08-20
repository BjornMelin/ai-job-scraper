# Archived AI/LLM ADRs

**Archive Date:** August 19, 2025  
**Supersession Status:** COMPLETE  

## Archived ADRs Summary

### ADR-002: LLM Provider and Model Selection

- **Status:** SUPERSEDED by ADR-009 (LLM Selection and Integration Strategy)
- **Reason:** OpenAI + Groq hybrid approach replaced by local Qwen3 models + cloud fallback
- **Key Change:** Cloud-primary strategy → Local-first with 98% local processing

### ADR-006: Agentic Workflows with LangGraph  

- **Status:** SUPERSEDED by ADR-002 (Minimal Implementation Guide)
- **Reason:** Complex agentic workflows replaced by library-first simple task processing
- **Key Change:** LangGraph orchestration → Direct RQ/Redis background jobs

## Migration Summary

### What Was Preserved

✅ **Core AI/LLM Requirements:** Accuracy, speed, and cost optimization maintained  
✅ **Performance Goals:** Low latency and high success rates carried forward  
✅ **Integration Patterns:** Clean service layer abstractions maintained  

### What Was Simplified  

🔄 **Architecture:** Complex agentic workflows → Simple task-based processing  
🔄 **Dependencies:** LangGraph + Cloud APIs → vLLM + local models  
🔄 **Cost Model:** $50/month cloud → $2.50/month local  
🔄 **Complexity:** Multi-agent coordination → Single model inference  

### Superseding ADRs

- **ADR-004:** Local AI Integration - Establishes vLLM + Qwen3 architecture
- **ADR-005:** Inference Stack - Defines model management and switching
- **ADR-006:** Hybrid Strategy - Cloud fallback for edge cases
- **ADR-009:** LLM Selection and Integration Strategy - Comprehensive replacement

## Historical Value

These archived ADRs represent important research into cloud-based AI approaches and agentic workflows that informed the design of the simplified local-first architecture that replaced them.

---

*These ADRs remain archived for historical reference and to understand the architectural evolution from cloud-dependent agentic systems to local-first inference approaches.*
