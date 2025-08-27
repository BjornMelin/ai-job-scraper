# MIGRATION VALIDATION FRAMEWORK

**Purpose:** Standardized validation procedures for SPEC-001 deletion operations  
**Scope:** Foundation demolition safety protocols and checkpoint validation  
**Reference:** ADR-001 Library-First Architecture compliance verification  

---

## DOCUMENTATION STANDARDS

### Timestamp Format

**Standard:** ISO 8601 with timezone  
**Examples:**

- `2025-08-27T14:30:45Z` (UTC)
- `2025-08-27T09:30:45-05:00` (Local with offset)

### Status Reporting Format

```
Status: [PENDING|IN_PROGRESS|COMPLETED|FAILED|ROLLBACK]
Started: YYYY-MM-DDTHH:MM:SSZ
Completed: YYYY-MM-DDTHH:MM:SSZ
Duration: MM minutes SS seconds
```

### Line Count Tracking Format

```
Lines Before: [number] lines
Lines After: [number] lines  
Reduction: [number] lines ([percentage]%)
Files Deleted: [count]
Directories Removed: [count]
```

### Error Reporting Template

```
ERROR: [Brief description]
Phase: [Phase number and name]
File: [Affected file path]
Issue: [Detailed error description]
Impact: [CRITICAL|HIGH|MEDIUM|LOW]
Action: [Required remediation]
Rollback: [YES/NO - if rollback needed]
```

---

## VALIDATION CHECKPOINT TEMPLATES

### Phase Start Checklist

```
PHASE [N]: [PHASE NAME] STARTING

Pre-Conditions:
□ Previous phase completed successfully
□ Working directory is clean  
□ Current branch is correct
□ Baseline metrics documented

Validation Commands:
□ git status (clean working tree)
□ find src -name "*.py" -exec wc -l {} + | tail -1 (line count)
□ python -c "import src; print('✅ Imports functional')" (import test)

Ready to Proceed: [YES/NO]
Started By: [Agent name]
Start Time: [Timestamp]
```

### Phase Completion Checklist

```
PHASE [N]: [PHASE NAME] COMPLETED

Post-Conditions:
□ All target files deleted successfully
□ No broken imports detected
□ Essential files preserved  
□ Line count reduction achieved
□ Checkpoint commit created

Validation Results:
□ Deletion verification: find [deleted_paths] -type f 2>/dev/null | wc -l = 0
□ Import integrity: python -c "import src; print('✅ Imports OK')"
□ Essential preservation: ls -la [preserved_files]
□ Performance check: [Any relevant performance validations]

Success Criteria Met: [YES/NO]
Completed By: [Agent name]  
End Time: [Timestamp]
Duration: [MM:SS]

Next Phase Ready: [YES/NO]
```

### Rollback Procedure Template

```
ROLLBACK INITIATED: [REASON]

Trigger Condition:
- Phase: [Failed phase]
- Issue: [Critical failure description]
- Impact: [System impact assessment]

Rollback Steps:
1. □ git stash (preserve any work)
2. □ git checkout [previous_checkpoint_commit]
3. □ git checkout -b rollback-[timestamp]
4. □ Verify system integrity
5. □ Document failure cause
6. □ Plan remediation approach

Rollback Validation:
□ System imports functional
□ Database integrity preserved  
□ No data loss occurred
□ Ready for retry after fix

Rollback Completed: [Timestamp]
Investigation Required: [YES/NO]
```

---

## SUCCESS/ERROR REPORTING FRAMEWORK

### Success Report Template

```
✅ SUCCESS: [Operation Description]

Achievement Summary:
- Operation: [Specific operation completed]
- Files Processed: [Count and list]
- Lines Reduced: [Before] → [After] ([Percentage]% reduction)
- Time Taken: [Duration]
- Quality Score: [Validation score]

Validation Passed:
□ All targeted files deleted
□ No import breakage
□ Performance maintained
□ Documentation updated
□ Checkpoint committed

Impact Assessment:
- Positive: [Benefits achieved]
- Risk Mitigation: [Risks eliminated]
- Next Phase Ready: [Readiness status]

Reported By: [Agent name]
Timestamp: [ISO 8601 timestamp]
```

### Error Report Template  

```
❌ ERROR: [Operation Description]

Failure Summary:
- Operation: [Attempted operation]
- Phase: [Current phase]
- Error Type: [IMPORT_FAILURE|FILE_SYSTEM|VALIDATION_FAILURE|OTHER]
- Severity: [CRITICAL|HIGH|MEDIUM|LOW]

Error Details:
- Command: [Failed command if applicable]
- Output: [Error output]
- File/Path: [Affected resources]
- Root Cause: [Analysis of failure]

Impact Assessment:
- System State: [Current system condition]  
- Data Integrity: [Data safety status]
- Rollback Required: [YES/NO]
- Deployment Risk: [Impact on deployment]

Remediation Plan:
1. [Step 1]
2. [Step 2]  
3. [Step 3]
4. [Validation step]

Reported By: [Agent name]
Timestamp: [ISO 8601 timestamp]
```

---

## IMPORT INTEGRITY VERIFICATION

### Standard Import Test Suite

```python
#!/usr/bin/env python3
"""
Standard import validation for SPEC-001 phases
Run after each deletion phase to verify system integrity
"""

import sys
import importlib

def validate_core_imports():
    """Test core system imports remain functional"""
    core_modules = [
        'src',
        'src.ai', 
        'src.services',
        'src.ui',
        'src.database',
        'src.models'
    ]
    
    results = {}
    for module in core_modules:
        try:
            importlib.import_module(module)
            results[module] = "✅ OK"
        except ImportError as e:
            results[module] = f"❌ FAIL: {e}"
        except Exception as e:
            results[module] = f"⚠️  ERROR: {e}"
    
    return results

def validate_essential_functionality():
    """Test essential functions still work"""
    tests = [
        ("Database Connection", "from src.database import engine; engine.connect()"),
        ("UI Components", "from src.ui import components"),  
        ("Service Layer", "from src.services import job_service"),
    ]
    
    results = {}
    for test_name, test_code in tests:
        try:
            exec(test_code)
            results[test_name] = "✅ FUNCTIONAL"
        except Exception as e:
            results[test_name] = f"❌ BROKEN: {e}"
    
    return results

if __name__ == "__main__":
    print("=== IMPORT INTEGRITY VALIDATION ===")
    
    import_results = validate_core_imports()
    for module, status in import_results.items():
        print(f"{module}: {status}")
    
    print("\n=== FUNCTIONALITY VALIDATION ===")
    func_results = validate_essential_functionality()
    for test, status in func_results.items():
        print(f"{test}: {status}")
    
    # Overall assessment
    failed_imports = [k for k, v in import_results.items() if "❌" in v]
    failed_functions = [k for k, v in func_results.items() if "❌" in v]
    
    if failed_imports or failed_functions:
        print(f"\n❌ VALIDATION FAILED: {len(failed_imports + failed_functions)} issues")
        sys.exit(1)
    else:
        print("\n✅ ALL VALIDATIONS PASSED")
        sys.exit(0)
```

### Usage Instructions

```bash
# Run after each phase deletion
python validation_suite.py

# Check specific imports  
python -c "import src.services; print('Services OK')"

# Validate file deletions
find src/coordination -type f 2>/dev/null | wc -l  # Should be 0 after phase 1

# Check remaining structure
find src -type f -name "*.py" | wc -l  # Track file count reduction
```

---

## LINE COUNT & METRICS TRACKING

### Automated Metrics Collection

```bash
#!/bin/bash
# collect_metrics.sh - Standardized metrics collection

echo "=== METRICS COLLECTION: $(date -Iseconds) ==="

echo "## Code Volume"
echo "Total Python files: $(find src -name "*.py" | wc -l)"
echo "Total lines: $(find src -name "*.py" -exec wc -l {} + | tail -1)"
echo "Total directories: $(find src -type d | wc -l)"

echo "## Largest Files (Top 10)"
find src -name "*.py" -exec wc -l {} + | sort -nr | head -10

echo "## Directory Sizes"
for dir in $(find src -maxdepth 2 -type d); do
    lines=$(find "$dir" -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}' 2>/dev/null || echo "0")
    files=$(find "$dir" -name "*.py" | wc -l)
    echo "$dir: $lines lines ($files files)"
done

echo "## Git Status"
echo "Branch: $(git branch --show-current)"
echo "Commits ahead: $(git rev-list --count HEAD ^origin/main 2>/dev/null || echo 'N/A')"
echo "Uncommitted changes: $(git status --porcelain | wc -l)"
```

### Progress Tracking Template

```
MIGRATION PROGRESS TRACKER

Phase 1 (Orchestration): [ ] 0% → Target: ~3,500 lines
Phase 2 (AI Infrastructure): [ ] 0% → Target: ~2,095 lines  
Phase 3 (Fragment System): [ ] 0% → Target: ~1,576 lines
Phase 4 (Scraping Services): [ ] 0% → Target: ~2,943 lines
Phase 5 (Cache Management): [ ] 0% → Target: ~506 lines

Current Status:
- Lines Remaining: [TBD] / 300 target
- Percentage Complete: [TBD]% 
- Estimated Completion: [TBD]
- Blockers: [None/List issues]

Quality Gates:
□ Import integrity maintained
□ Functionality preserved  
□ Documentation updated
□ Safety backups current
```

---

## EMERGENCY PROCEDURES

### Critical Failure Response

1. **STOP** all deletion operations immediately
2. **ASSESS** system state and data integrity  
3. **ROLLBACK** to last known good state if needed
4. **INVESTIGATE** root cause before resuming
5. **DOCUMENT** failure for process improvement

### Emergency Contacts & Resources

- **Safety Backup Branch:** `safety-backup-[timestamp]`  
- **Emergency Recovery:** See MIGRATION-LOG.md rollback procedures
- **Validation Commands:** Run validation_suite.py for system check
- **Documentation:** All procedures documented in this framework

### Recovery Validation Checklist

□ System starts without errors
□ Database connections work
□ Core imports functional  
□ No data corruption detected
□ Ready to retry failed operation

---

*This framework provides standardized procedures for safe, auditable migration execution*  
*All deletion agents must follow these validation protocols during SPEC-001*
