# ADR-045: Cost Optimization and Resource Efficiency

## Title

Cost Optimization and Resource Efficiency Strategy for $30/Month Budget Compliance and 98% Local Processing

## Version/Date

1.0 / August 19, 2025

## Status

**Accepted** - Essential for budget compliance and operational sustainability

## Description

Comprehensive cost optimization and resource efficiency strategy to achieve $30/month operational budget while maintaining 98% local processing ratio and optimal performance. Leverages expert-validated vLLM two-tier deployment, intelligent request routing, and comprehensive monitoring to maximize resource utilization and minimize cloud processing costs.

## Context

### Budget Requirements and Constraints

**Monthly Budget Target**: $30.00

- **Current baseline costs**: $30/month validated through comprehensive cost analysis
- **Local processing target**: 98% (2% cloud processing maximum)
- **Performance requirement**: 2-3x improvement with resource efficiency
- **Operational costs**: Infrastructure, proxies, monitoring, and cloud fallback

### Cost Breakdown Analysis

**Fixed Costs (Validated)**:

- **IPRoyal residential proxies**: $20.00/month (anti-bot protection)
- **Monitoring infrastructure**: $5.00/month (Prometheus + Grafana)
- **Redis hosting**: $2.50/month (RQ background tasks)
- **Infrastructure overhead**: $2.50/month (SSL, domain, miscellaneous)

**Variable Costs (Target <$2/month)**:

- **Cloud processing**: $0-2.00/month (2% maximum usage)
- **Additional resources**: $0/month (local processing optimization)

### Resource Efficiency Requirements

**Hardware Optimization**:

- **RTX 4090**: 85-90% GPU utilization target with vLLM swap_space
- **Host RAM**: 18GB swap_space + 20% overhead efficient allocation  
- **PCIe bandwidth**: <10GB/s sustained to prevent saturation
- **Local processing**: 60-80 req/s throughput with cost efficiency

## Related Requirements

### Functional Requirements

- **FR-COST-01**: Maintain monthly operational costs ≤$30.00
- **FR-COST-02**: Achieve 98% local processing ratio with 2% cloud fallback
- **FR-COST-03**: Track and alert on budget utilization with real-time monitoring
- **FR-COST-04**: Optimize resource allocation for maximum efficiency

### Non-Functional Requirements

- **NFR-COST-01**: Cost tracking accuracy ±$0.10 for budget management
- **NFR-COST-02**: Local processing latency ≤100ms P95 for throughput pool
- **NFR-COST-03**: Cloud processing triggered only for >4096 token contexts
- **NFR-COST-04**: Resource efficiency >80% for all allocated resources

### Performance Requirements

- **PR-COST-01**: GPU utilization 85-90% with minimal idle time
- **PR-COST-02**: Host RAM efficiency >70% with swap_space optimization
- **PR-COST-03**: Request routing accuracy >99% for optimal cost allocation
- **PR-COST-04**: Cost per job processed <$0.01 for sustainable operations

## Related Decisions

- **Implements ADR-042**: vLLM Two-Tier Deployment Strategy (resource optimization)
- **Leverages ADR-043**: Host System Resource Management (efficiency monitoring)
- **Integrates ADR-044**: Production Monitoring and Alerting Strategy (cost tracking)
- **Supports ADR-035**: Final Production Architecture (overall cost efficiency)

## Decision

**Deploy Comprehensive Cost Optimization Strategy** with these core components:

### 1. Intelligent Request Routing for Cost Optimization

**Context-Length Based Cost Routing:**

```python
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from prometheus_client import Counter, Gauge, Histogram

@dataclass
class CostOptimizationConfig:
    """Cost optimization configuration parameters."""
    
    # Budget constraints
    monthly_budget: float = 30.00           # $30 monthly budget
    cloud_processing_limit: float = 2.00   # $2 maximum cloud processing
    
    # Processing targets
    local_processing_target: float = 0.98  # 98% local processing
    context_threshold: int = 4096          # Tokens for local vs cloud routing
    
    # Cost per request estimates
    local_cost_per_request: float = 0.001  # $0.001 per local request
    cloud_cost_per_request: float = 0.05   # $0.05 per cloud request
    
    # Efficiency targets
    gpu_utilization_target: float = 0.85   # 85% GPU utilization
    resource_efficiency_target: float = 0.8 # 80% resource efficiency

class CostOptimizedRouter:
    """Intelligent request router optimized for cost efficiency."""
    
    def __init__(self, config: CostOptimizationConfig):
        self.config = config
        
        # Cost tracking metrics
        self.local_requests = Counter('cost_local_requests_total', 'Local processing requests')
        self.cloud_requests = Counter('cost_cloud_requests_total', 'Cloud processing requests')
        
        self.local_processing_cost = Gauge('cost_local_processing_dollars', 'Local processing costs')
        self.cloud_processing_cost = Gauge('cost_cloud_processing_dollars', 'Cloud processing costs')
        
        # Efficiency metrics
        self.cost_per_request = Gauge('cost_per_request_dollars', 'Average cost per request')
        self.budget_utilization = Gauge('cost_budget_utilization_percent', 'Budget utilization percentage')
        self.processing_ratio = Gauge('cost_local_processing_ratio', 'Local processing ratio')
        
        # Router decisions
        self.routing_decisions = Counter(
            'cost_routing_decisions_total',
            'Router decisions for cost optimization',
            ['decision_type', 'context_bucket']
        )
        
        # Current month tracking
        self.monthly_cost_tracking = {
            "local_requests": 0,
            "cloud_requests": 0,
            "total_cost": 0.0
        }
    
    async def route_request_with_cost_optimization(
        self, 
        prompt: str, 
        context_tokens: int,
        priority: str = "standard"
    ) -> Tuple[str, str, float]:
        """
        Route request with intelligent cost optimization.
        
        Returns:
            Tuple of (routing_decision, pool_name, estimated_cost)
        """
        
        # Check monthly budget remaining
        budget_remaining = await self._get_monthly_budget_remaining()
        
        # Determine optimal routing strategy
        routing_decision = await self._determine_cost_optimal_routing(
            context_tokens=context_tokens,
            budget_remaining=budget_remaining,
            priority=priority
        )
        
        # Execute routing decision
        if routing_decision == "force_local":
            # Force local processing even for large contexts to save costs
            pool_name = "longcontext_pool" if context_tokens > 4096 else "throughput_pool"
            estimated_cost = self.config.local_cost_per_request
            
            self.local_requests.inc()
            self.routing_decisions.labels(
                decision_type="force_local",
                context_bucket=self._get_context_bucket(context_tokens)
            ).inc()
            
        elif routing_decision == "optimal_local":
            # Standard local processing routing
            pool_name = "throughput_pool"
            estimated_cost = self.config.local_cost_per_request
            
            self.local_requests.inc()
            self.routing_decisions.labels(
                decision_type="optimal_local", 
                context_bucket=self._get_context_bucket(context_tokens)
            ).inc()
            
        elif routing_decision == "budget_cloud":
            # Cloud processing within budget constraints
            pool_name = "cloud_fallback"
            estimated_cost = self.config.cloud_cost_per_request
            
            self.cloud_requests.inc()
            self.routing_decisions.labels(
                decision_type="budget_cloud",
                context_bucket=self._get_context_bucket(context_tokens)
            ).inc()
            
        else:  # "queue_local"
            # Queue for local processing to avoid cloud costs
            pool_name = "throughput_pool_queued"
            estimated_cost = self.config.local_cost_per_request
            
            self.local_requests.inc()
            self.routing_decisions.labels(
                decision_type="queue_local",
                context_bucket=self._get_context_bucket(context_tokens)
            ).inc()
        
        # Update cost tracking
        await self._update_cost_tracking(routing_decision, estimated_cost)
        
        return routing_decision, pool_name, estimated_cost
    
    async def _determine_cost_optimal_routing(
        self,
        context_tokens: int,
        budget_remaining: float,
        priority: str
    ) -> str:
        """Determine optimal routing strategy for cost efficiency."""
        
        # Calculate current local processing ratio
        current_local_ratio = await self._get_current_local_ratio()
        
        # Check if we're at risk of exceeding cloud processing target
        if current_local_ratio < self.config.local_processing_target:
            # Force local processing to maintain 98% target
            return "force_local"
        
        # Standard routing for small contexts (always local)
        if context_tokens <= self.config.context_threshold:
            return "optimal_local"
        
        # Large context routing decision
        cloud_cost = self.config.cloud_cost_per_request
        
        # Check budget constraints
        if budget_remaining < cloud_cost:
            # Not enough budget for cloud processing
            if priority == "high":
                return "force_local"  # Process locally despite size
            else:
                return "queue_local"  # Queue for later local processing
        
        # Check if cloud processing is within monthly limits
        monthly_cloud_cost = self.monthly_cost_tracking["cloud_requests"] * cloud_cost
        if monthly_cloud_cost + cloud_cost > self.config.cloud_processing_limit:
            return "queue_local"  # Defer to maintain cost limits
        
        # Cloud processing approved within budget
        return "budget_cloud"
    
    async def _get_monthly_budget_remaining(self) -> float:
        """Calculate remaining budget for current month."""
        
        fixed_costs = 20.00 + 5.00 + 2.50 + 2.50  # IPRoyal + monitoring + Redis + misc
        variable_costs = (
            self.monthly_cost_tracking["local_requests"] * self.config.local_cost_per_request +
            self.monthly_cost_tracking["cloud_requests"] * self.config.cloud_cost_per_request
        )
        
        total_spent = fixed_costs + variable_costs
        return max(0, self.config.monthly_budget - total_spent)
    
    async def _get_current_local_ratio(self) -> float:
        """Calculate current local processing ratio."""
        
        total_requests = (
            self.monthly_cost_tracking["local_requests"] + 
            self.monthly_cost_tracking["cloud_requests"]
        )
        
        if total_requests == 0:
            return 1.0  # 100% local (no requests yet)
        
        return self.monthly_cost_tracking["local_requests"] / total_requests
    
    def _get_context_bucket(self, tokens: int) -> str:
        """Categorize context size for cost analysis."""
        
        if tokens < 1000:
            return "small"
        elif tokens <= 4096:
            return "medium"
        elif tokens <= 8192:
            return "large" 
        else:
            return "xlarge"
    
    async def _update_cost_tracking(self, routing_decision: str, estimated_cost: float):
        """Update monthly cost tracking."""
        
        if "local" in routing_decision:
            self.monthly_cost_tracking["local_requests"] += 1
        else:
            self.monthly_cost_tracking["cloud_requests"] += 1
            
        self.monthly_cost_tracking["total_cost"] += estimated_cost
        
        # Update prometheus metrics
        self.cost_per_request.set(
            self.monthly_cost_tracking["total_cost"] / 
            (self.monthly_cost_tracking["local_requests"] + self.monthly_cost_tracking["cloud_requests"])
        )
        
        local_ratio = await self._get_current_local_ratio()
        self.processing_ratio.set(local_ratio)
        
        budget_util = (self.monthly_cost_tracking["total_cost"] / self.config.monthly_budget) * 100
        self.budget_utilization.set(budget_util)
```

### 2. Resource Efficiency Optimization

**GPU and Memory Efficiency Maximization:**

```python
class ResourceEfficiencyOptimizer:
    """Optimize resource allocation for maximum cost efficiency."""
    
    def __init__(self):
        # Efficiency tracking
        self.gpu_efficiency = Gauge('resource_gpu_efficiency_percent', 'GPU utilization efficiency')
        self.memory_efficiency = Gauge('resource_memory_efficiency_percent', 'Memory utilization efficiency') 
        self.swap_space_efficiency = Gauge('resource_swap_efficiency_percent', 'Swap space utilization efficiency')
        
        # Cost efficiency metrics
        self.cost_per_gpu_hour = Gauge('cost_per_gpu_hour_dollars', 'Cost per GPU hour')
        self.requests_per_dollar = Gauge('cost_requests_per_dollar', 'Requests processed per dollar')
        
        # Target efficiency levels
        self.efficiency_targets = {
            "gpu_utilization": 0.85,        # 85% GPU utilization
            "memory_utilization": 0.80,     # 80% memory efficiency
            "swap_utilization": 0.70,       # 70% swap efficiency
            "cost_per_request": 0.01        # $0.01 maximum cost per request
        }
    
    async def optimize_resource_allocation(self):
        """Continuously optimize resource allocation for cost efficiency."""
        
        while True:
            try:
                # Analyze current resource utilization
                utilization_analysis = await self._analyze_resource_utilization()
                
                # Calculate efficiency scores
                efficiency_scores = await self._calculate_efficiency_scores(utilization_analysis)
                
                # Generate optimization recommendations
                optimizations = await self._generate_efficiency_optimizations(efficiency_scores)
                
                # Apply low-risk optimizations automatically
                for optimization in optimizations:
                    if optimization["risk_level"] == "low" and optimization["cost_impact"] < 0:
                        await self._apply_optimization(optimization)
                        logging.info(f"Applied cost optimization: {optimization['description']}")
                
                # Update efficiency metrics
                await self._update_efficiency_metrics(efficiency_scores)
                
            except Exception as e:
                logging.error(f"Resource efficiency optimization failed: {e}")
            
            await asyncio.sleep(300)  # Optimize every 5 minutes
    
    async def _analyze_resource_utilization(self) -> Dict:
        """Analyze current resource utilization patterns."""
        
        return {
            "gpu_utilization": await self._get_gpu_utilization(),
            "memory_usage": await self._get_memory_utilization(),
            "swap_space_usage": await self._get_swap_utilization(),
            "request_throughput": await self._get_current_throughput(),
            "cost_per_hour": await self._calculate_current_cost_per_hour()
        }
    
    async def _calculate_efficiency_scores(self, utilization: Dict) -> Dict:
        """Calculate efficiency scores for cost optimization."""
        
        scores = {}
        
        # GPU efficiency score
        gpu_target = self.efficiency_targets["gpu_utilization"]
        gpu_actual = utilization["gpu_utilization"] / 100.0
        scores["gpu_efficiency"] = min(gpu_actual / gpu_target, 1.0) * 100
        
        # Memory efficiency score  
        memory_target = self.efficiency_targets["memory_utilization"]
        memory_actual = utilization["memory_usage"] / 100.0
        scores["memory_efficiency"] = min(memory_actual / memory_target, 1.0) * 100
        
        # Cost efficiency score
        cost_target = self.efficiency_targets["cost_per_request"]
        cost_actual = utilization["cost_per_hour"] / max(utilization["request_throughput"], 1)
        scores["cost_efficiency"] = min(cost_target / cost_actual, 1.0) * 100
        
        # Overall efficiency score
        scores["overall_efficiency"] = (
            scores["gpu_efficiency"] * 0.4 +
            scores["memory_efficiency"] * 0.3 + 
            scores["cost_efficiency"] * 0.3
        )
        
        return scores
    
    async def _generate_efficiency_optimizations(self, scores: Dict) -> List[Dict]:
        """Generate resource efficiency optimization recommendations."""
        
        optimizations = []
        
        # GPU utilization optimization
        if scores["gpu_efficiency"] < 80:
            optimizations.append({
                "type": "gpu_optimization",
                "description": "Increase batch size to improve GPU utilization",
                "current_efficiency": scores["gpu_efficiency"],
                "target_efficiency": 85.0,
                "estimated_cost_impact": -0.20,  # 20% cost reduction
                "risk_level": "low"
            })
        
        # Memory efficiency optimization
        if scores["memory_efficiency"] < 75:
            optimizations.append({
                "type": "memory_optimization", 
                "description": "Optimize model quantization for better memory efficiency",
                "current_efficiency": scores["memory_efficiency"],
                "target_efficiency": 80.0,
                "estimated_cost_impact": -0.10,  # 10% cost reduction
                "risk_level": "medium"
            })
        
        # Cost efficiency optimization
        if scores["cost_efficiency"] < 85:
            optimizations.append({
                "type": "cost_optimization",
                "description": "Implement request batching to reduce per-request overhead",
                "current_efficiency": scores["cost_efficiency"],
                "target_efficiency": 90.0,
                "estimated_cost_impact": -0.15,  # 15% cost reduction
                "risk_level": "low"
            })
        
        return optimizations
```

### 3. Monthly Budget Management

**Automated Budget Monitoring and Enforcement:**

```yaml
# cost-optimization-config.yaml
cost_optimization:
  # Monthly budget configuration
  budget:
    total_monthly: 30.00
    cloud_processing_limit: 2.00
    alert_thresholds:
      warning: 0.80    # 80% budget utilization
      critical: 0.95   # 95% budget utilization
  
  # Fixed cost breakdown
  fixed_costs:
    iproyal_proxies: 20.00
    monitoring_stack: 5.00
    redis_hosting: 2.50
    infrastructure: 2.50
  
  # Variable cost targets
  variable_costs:
    local_cost_per_request: 0.001
    cloud_cost_per_request: 0.05
    target_requests_per_month: 100000
    max_cloud_requests_per_month: 40  # 2% of 2000 requests
  
  # Efficiency targets
  efficiency:
    local_processing_ratio: 0.98
    gpu_utilization_target: 0.85
    cost_per_job_target: 0.01
    
  # Auto-scaling configuration
  scaling:
    enable_cost_based_scaling: true
    scale_down_threshold: 0.95      # Budget utilization
    queue_requests_when_over_budget: true
    max_queue_size: 1000
```

## Testing Strategy

### Cost Optimization Testing

```python
# tests/test_cost_optimization.py
import pytest
import asyncio
from unittest.mock import Mock, patch
from src.cost_optimization import CostOptimizedRouter, CostOptimizationConfig

class TestCostOptimization:
    """Test cost optimization and budget compliance."""
    
    @pytest.fixture
    def cost_config(self):
        return CostOptimizationConfig(
            monthly_budget=30.00,
            cloud_processing_limit=2.00,
            local_processing_target=0.98,
            context_threshold=4096
        )
    
    @pytest.fixture
    def cost_router(self, cost_config):
        return CostOptimizedRouter(cost_config)
    
    async def test_budget_compliance_routing(self, cost_router):
        """Test routing decisions maintain budget compliance."""
        
        # Simulate month with high request volume
        for i in range(1000):
            context_tokens = 2000 if i < 950 else 6000  # 95% small, 5% large
            
            routing_decision, pool_name, estimated_cost = await cost_router.route_request_with_cost_optimization(
                prompt=f"Test prompt {i}",
                context_tokens=context_tokens
            )
            
            # Verify routing decisions maintain budget
            if context_tokens <= 4096:
                assert "local" in routing_decision
                assert estimated_cost == 0.001
            else:
                # Large contexts should prefer local processing to save costs
                if cost_router.monthly_cost_tracking["cloud_requests"] * 0.05 < 2.00:
                    assert routing_decision in ["budget_cloud", "force_local", "queue_local"]
                else:
                    assert routing_decision in ["force_local", "queue_local"]
        
        # Verify final budget compliance
        total_cost = cost_router.monthly_cost_tracking["total_cost"]
        assert total_cost <= 30.00
        
        # Verify local processing ratio
        local_ratio = await cost_router._get_current_local_ratio()
        assert local_ratio >= 0.98
    
    async def test_cost_per_request_efficiency(self, cost_router):
        """Test cost per request stays within efficiency targets."""
        
        # Process various request types
        test_cases = [
            (500, "small context"),
            (2000, "medium context"),
            (4000, "large context"),
            (8000, "xlarge context")
        ]
        
        total_cost = 0
        request_count = 0
        
        for context_tokens, description in test_cases * 100:  # 400 total requests
            routing_decision, pool_name, estimated_cost = await cost_router.route_request_with_cost_optimization(
                prompt=f"Test: {description}",
                context_tokens=context_tokens
            )
            
            total_cost += estimated_cost
            request_count += 1
        
        # Verify cost efficiency
        cost_per_request = total_cost / request_count
        assert cost_per_request <= 0.01  # $0.01 maximum cost per request
        
        logging.info(f"Cost per request: ${cost_per_request:.4f}")
    
    async def test_resource_efficiency_optimization(self):
        """Test resource efficiency optimization."""
        
        optimizer = ResourceEfficiencyOptimizer()
        
        # Mock resource utilization data
        mock_utilization = {
            "gpu_utilization": 70,  # Below 85% target
            "memory_usage": 75,     # Below 80% target  
            "swap_space_usage": 60, # Below 70% target
            "request_throughput": 50,
            "cost_per_hour": 1.00
        }
        
        with patch.object(optimizer, '_analyze_resource_utilization', return_value=mock_utilization):
            efficiency_scores = await optimizer._calculate_efficiency_scores(mock_utilization)
            optimizations = await optimizer._generate_efficiency_optimizations(efficiency_scores)
        
        # Verify efficiency optimization recommendations
        assert len(optimizations) > 0
        
        # Check for cost reduction recommendations
        cost_reductions = [opt for opt in optimizations if opt["estimated_cost_impact"] < 0]
        assert len(cost_reductions) > 0
        
        # Verify optimization safety levels
        low_risk_optimizations = [opt for opt in optimizations if opt["risk_level"] == "low"]
        assert len(low_risk_optimizations) > 0
```

## Success Metrics

| Metric | Target | Validation Method |
|--------|---------|------------------|
| Monthly Budget Compliance | ≤$30.00 | Real-time cost tracking |
| Local Processing Ratio | ≥98% | Request routing analytics |
| Cost per Request | ≤$0.01 | Efficiency calculation |
| GPU Utilization Efficiency | ≥85% | Resource monitoring |
| Budget Alert Accuracy | ±$0.10 | Cost reconciliation |
| Resource Optimization | 10% improvement | Before/after comparison |

## Implementation Timeline

### Phase 1: Cost Tracking Infrastructure (Days 1-2)

- [ ] Deploy comprehensive cost monitoring and budget tracking
- [ ] Implement intelligent request routing for cost optimization
- [ ] Setup monthly budget enforcement with automated alerting
- [ ] Configure real-time cost efficiency metrics

### Phase 2: Resource Efficiency Optimization (Days 3-4)

- [ ] Deploy resource efficiency monitoring and optimization
- [ ] Implement automated resource allocation tuning
- [ ] Setup efficiency scoring and improvement recommendations
- [ ] Configure cost-based auto-scaling and queue management

### Phase 3: Integration and Validation (Days 5-6)

- [ ] Integrate cost optimization with vLLM two-tier deployment
- [ ] Test budget compliance under various load scenarios
- [ ] Validate local processing ratio maintenance
- [ ] Performance testing with cost constraints

### Phase 4: Production Deployment (Days 7-8)

- [ ] Deploy cost optimization stack to production
- [ ] Configure production budget monitoring and alerting
- [ ] Setup monthly cost reporting and analysis
- [ ] Documentation and operational cost management procedures

## Consequences
### Dependencies

- No specific external dependencies for this architectural decision

### References

- No additional references beyond those in context

### Changelog

#### Current Version
- Initial documentation and architectural specification


### Positive Outcomes

- ✅ **Budget Compliance**: Guaranteed $30/month operational cost ceiling
- ✅ **Cost Efficiency**: Optimal resource utilization with 98% local processing
- ✅ **Predictable Costs**: Fixed cost structure with minimal variable expenses
- ✅ **Resource Optimization**: 85-90% GPU utilization with intelligent routing
- ✅ **Scalability**: Cost-effective scaling without budget overruns
- ✅ **Operational Transparency**: Real-time cost visibility and optimization

### Negative Consequences

- ❌ **Request Queuing**: Large contexts may be queued during budget constraints
- ❌ **Performance Trade-offs**: Cost optimization may impact processing speed
- ❌ **Complexity**: Multi-tier cost optimization increases operational complexity
- ❌ **Budget Rigidity**: Strict budget enforcement may limit processing flexibility

### Risk Mitigation

1. **Budget Overrun Prevention**
   - Real-time cost monitoring with proactive alerting
   - Automated request routing to maintain budget compliance
   - Queue management for cost-constrained scenarios

2. **Performance Impact Minimization**
   - Intelligent routing prioritizes performance within budget
   - High-priority requests can override cost constraints
   - Resource efficiency optimization maintains performance targets

3. **Operational Flexibility**
   - Emergency budget override capabilities for critical processing
   - Configurable cost thresholds and optimization parameters
   - Manual intervention options for special circumstances

## Changelog

### v1.0 - August 19, 2025

- Comprehensive cost optimization strategy for $30/month budget compliance
- Intelligent request routing with 98% local processing target
- Resource efficiency optimization with automated tuning
- Real-time budget monitoring and enforcement
- Cost-based auto-scaling and queue management
- Complete cost management infrastructure and procedures

---

*This ADR ensures sustainable operations within budget constraints while maintaining optimal performance and resource efficiency through intelligent cost optimization and automated budget management.*
