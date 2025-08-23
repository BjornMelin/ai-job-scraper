# AI Quality Testing Implementation Guide

**Related ADRs:**
- **ADR-035** (AI Processing Quality Assurance) - PROOF methodology and framework
- **ADR-034** (Strategic LLM Configuration Decisions) - WHY testing these decisions
- **ADR-004** (Local AI Integration Architecture) - HOW integration should be tested

## Overview

This guide provides detailed testing implementation for validating the AI processing quality assurance framework defined in ADR-035. The testing suite validates integration, performance, accuracy, and reliability of Qwen3-4B-Instruct-2507-FP8 with vLLM integration.

## Comprehensive Testing Suite

### Core Testing Framework

```python
import pytest
import time
import asyncio
import logging
import json
import statistics
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys
import os

# Test configuration and fixtures
@pytest.fixture(scope="session")
def ai_processor():
    """Initialize AI processor for testing session."""
    from local_ai_processor import ComprehensiveLocalAIProcessor
    
    processor = ComprehensiveLocalAIProcessor()
    
    # Verify processor is ready
    health = processor.health_check()
    if health["status"] != "healthy":
        pytest.skip(f"AI processor not healthy: {health['status']}")
    
    return processor

@pytest.fixture(scope="session") 
def test_job_schema():
    """Standard job extraction schema for testing."""
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "company": {"type": "string"},
            "location": {"type": ["string", "null"]},
            "salary_min": {"type": ["integer", "null"]},
            "salary_max": {"type": ["integer", "null"]},
            "skills": {"type": "array", "items": {"type": "string"}},
            "description": {"type": "string"},
            "requirements": {"type": "array", "items": {"type": "string"}},
            "benefits": {"type": ["string", "null"]},
            "employment_type": {"type": ["string", "null"]},
            "experience_level": {"type": ["string", "null"]}
        },
        "required": ["title", "company", "description"]
    }

@pytest.fixture(scope="session")
def benchmark_dataset():
    """Load comprehensive benchmark dataset for accuracy testing."""
    return [
        {
            "id": "test_001",
            "raw_content": """Senior Software Engineer - Remote
TechCorp Inc.
Looking for experienced Python developer with React and AWS skills.
Salary: $120,000 - $150,000
Full-time position with comprehensive benefits package.
Requirements: 5+ years Python, React, AWS experience.""",
            "ground_truth": {
                "title": "Senior Software Engineer",
                "company": "TechCorp Inc.",
                "location": "Remote",
                "salary_min": 120000,
                "salary_max": 150000,
                "skills": ["Python", "React", "AWS"],
                "description": "Looking for experienced Python developer with React and AWS skills.",
                "requirements": ["5+ years Python", "React", "AWS experience"],
                "employment_type": "Full-time",
                "experience_level": "Senior"
            }
        },
        {
            "id": "test_002",
            "raw_content": """Data Scientist
StartupCorp
Machine Learning Engineer position in San Francisco.
$90k-$130k + equity
Part-time/Contract available
Must have Python, TensorFlow, SQL experience.""",
            "ground_truth": {
                "title": "Data Scientist",
                "company": "StartupCorp", 
                "location": "San Francisco",
                "salary_min": 90000,
                "salary_max": 130000,
                "skills": ["Python", "TensorFlow", "SQL", "Machine Learning"],
                "description": "Machine Learning Engineer position in San Francisco.",
                "requirements": ["Python", "TensorFlow", "SQL experience"],
                "employment_type": "Part-time/Contract"
            }
        },
        {
            "id": "test_003", 
            "raw_content": """DevOps Engineer
CloudCompany Ltd.
Kubernetes, Docker, AWS expertise required.
Remote-friendly position.
Competitive salary and benefits.
3+ years infrastructure experience needed.""",
            "ground_truth": {
                "title": "DevOps Engineer",
                "company": "CloudCompany Ltd.",
                "location": "Remote-friendly",
                "skills": ["Kubernetes", "Docker", "AWS"],
                "description": "Kubernetes, Docker, AWS expertise required.",
                "requirements": ["3+ years infrastructure experience needed"],
                "employment_type": null
            }
        },
        # Additional test cases for comprehensive coverage
    ] * 20  # Replicate for larger test dataset

class TestAIQualityAssurance:
    """
    Comprehensive AI quality assurance testing suite implementing ADR-035 methodology.
    
    Tests integration, performance, accuracy, and reliability of local AI processing
    with systematic validation of ADR-034 strategic decisions and ADR-004 architecture.
    """
    
    def setup_method(self):
        """Setup for each test method."""
        self.quality_thresholds = {
            "accuracy_minimum": 0.95,
            "latency_maximum": 1.0,
            "memory_maximum_gb": 4.0,
            "reliability_minimum": 0.95,
            "throughput_minimum": 45
        }
        self.test_results = {}
    
    # Integration Testing Suite
    @pytest.mark.integration
    def test_vllm_fp8_initialization_comprehensive(self, ai_processor):
        """
        Comprehensive vLLM FP8 initialization test validating ADR-034 configuration.
        
        Validates:
        - FP8 quantization is active and working
        - Model configuration matches strategic decisions
        - Hardware compatibility with RTX 4090 Laptop GPU
        - Memory optimization is functioning
        """
        
        # Verify processor is initialized
        assert ai_processor is not None
        
        # Health check validation
        health = ai_processor.health_check()
        assert health["status"] == "healthy", f"Processor not healthy: {health['status']}"
        assert health["model_loaded"] is True
        assert health["fp8_quantization"] is True
        
        # Model configuration validation
        assert "Qwen3-4B-Instruct-2507-FP8" in ai_processor.model_name
        
        # Memory efficiency validation (FP8 quantization benefit)
        memory_stats = ai_processor.get_memory_stats()
        assert memory_stats["model_memory_gb"] <= 1.5, f"Model memory too high: {memory_stats['model_memory_gb']}GB"
        assert memory_stats["status"] == "optimal"
        assert memory_stats["quantization"] == "fp8"
        
        # GPU utilization validation
        assert memory_stats["gpu_utilization_percent"] >= 80, "GPU utilization too low"
        
        # Hardware compatibility check
        assert "RTX 4090" in memory_stats.get("hardware", "")
        
        logging.info("✅ vLLM FP8 initialization validation: PASSED")
        
        self.test_results["integration_test"] = {
            "status": "PASSED",
            "health_check": health,
            "memory_stats": memory_stats
        }
    
    # Accuracy Validation Testing Suite
    @pytest.mark.accuracy
    @pytest.mark.asyncio
    async def test_extraction_accuracy_comprehensive(self, ai_processor, test_job_schema, benchmark_dataset):
        """
        Comprehensive accuracy validation against benchmark dataset.
        
        Validates extraction accuracy meets 95%+ threshold per ADR-035 requirements
        with statistical analysis and detailed error reporting.
        """
        
        accuracy_scores = []
        extraction_details = []
        failed_extractions = []
        
        # Test on comprehensive benchmark dataset
        for job_sample in benchmark_dataset[:50]:  # Test subset for CI efficiency
            job_id = job_sample["id"]
            content = job_sample["raw_content"] 
            expected = job_sample["ground_truth"]
            
            try:
                # Extract job data
                start_time = time.monotonic()
                extracted = await ai_processor.extract_jobs(content, test_job_schema)
                end_time = time.monotonic()
                
                extraction_time = end_time - start_time
                
                if "_extraction_success" in extracted and extracted["_extraction_success"]:
                    # Calculate accuracy for this extraction
                    accuracy = self._calculate_extraction_accuracy(extracted, expected)
                    accuracy_scores.append(accuracy)
                    
                    extraction_details.append({
                        "job_id": job_id,
                        "accuracy": accuracy,
                        "extraction_time": extraction_time,
                        "extracted": extracted,
                        "expected": expected
                    })
                    
                    logging.debug(f"Job {job_id}: Accuracy {accuracy:.3f}, Time {extraction_time:.3f}s")
                    
                else:
                    failed_extractions.append({
                        "job_id": job_id,
                        "error": extracted.get("error", "Unknown error"),
                        "content_preview": content[:100] + "..."
                    })
                    
            except Exception as e:
                failed_extractions.append({
                    "job_id": job_id,
                    "error": f"Exception: {str(e)}",
                    "content_preview": content[:100] + "..."
                })
                logging.error(f"❌ Extraction failed for job {job_id}: {e}")
        
        # Statistical analysis
        if accuracy_scores:
            avg_accuracy = statistics.mean(accuracy_scores)
            std_accuracy = statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0.0
            min_accuracy = min(accuracy_scores)
            max_accuracy = max(accuracy_scores)
            
            # Validate accuracy threshold
            assert avg_accuracy >= self.quality_thresholds["accuracy_minimum"], \
                f"Average accuracy {avg_accuracy:.3f} below {self.quality_thresholds['accuracy_minimum']:.1%} threshold"
            
            # Additional quality checks
            assert min_accuracy >= 0.8, f"Minimum accuracy {min_accuracy:.3f} too low"
            assert len(failed_extractions) / len(benchmark_dataset[:50]) <= 0.05, \
                f"Too many failed extractions: {len(failed_extractions)}/{len(benchmark_dataset[:50])}"
            
            logging.info(f"✅ Accuracy validation: {avg_accuracy:.1%} (±{std_accuracy:.2f}) on {len(accuracy_scores)} samples")
            logging.info(f"📊 Range: {min_accuracy:.3f} - {max_accuracy:.3f}")
            
            self.test_results["accuracy_test"] = {
                "status": "PASSED",
                "average_accuracy": avg_accuracy,
                "std_deviation": std_accuracy,
                "sample_size": len(accuracy_scores),
                "failed_extractions": len(failed_extractions),
                "threshold_met": avg_accuracy >= self.quality_thresholds["accuracy_minimum"],
                "detailed_results": extraction_details[:5]  # Sample results
            }
            
        else:
            pytest.fail("No successful extractions to validate accuracy")
    
    # Performance Testing Suite
    @pytest.mark.performance  
    @pytest.mark.asyncio
    async def test_inference_latency_comprehensive(self, ai_processor, test_job_schema):
        """
        Comprehensive latency testing validating <1.0s inference requirement.
        
        Tests various content lengths and provides statistical analysis
        of inference performance under different conditions.
        """
        
        latency_measurements = []
        test_cases = [
            ("short", "Software Engineer at TechCorp. Python required." * 20),
            ("medium", "Senior Developer position with cloud experience." * 100), 
            ("optimal", "Full-stack role with React, Node.js, AWS." * 200),
            ("maximum", "Lead Engineer with extensive requirements." * 300)  # ~8K tokens
        ]
        
        for case_name, content in test_cases:
            case_latencies = []
            
            # Multiple measurements for statistical validity
            for iteration in range(10):
                start_time = time.monotonic()
                result = await ai_processor.extract_jobs(content, test_job_schema)
                end_time = time.monotonic()
                
                latency = end_time - start_time
                case_latencies.append(latency)
                
                # Verify extraction succeeded
                assert "_extraction_success" in result and result["_extraction_success"], \
                    f"Extraction failed for {case_name} case iteration {iteration}"
            
            avg_latency = statistics.mean(case_latencies)
            latency_measurements.extend(case_latencies)
            
            logging.debug(f"{case_name} content - Avg latency: {avg_latency:.3f}s")
        
        # Overall latency validation
        overall_avg_latency = statistics.mean(latency_measurements)
        latency_std = statistics.stdev(latency_measurements)
        max_latency = max(latency_measurements)
        min_latency = min(latency_measurements)
        
        # Threshold validation
        assert overall_avg_latency < self.quality_thresholds["latency_maximum"], \
            f"Average latency {overall_avg_latency:.3f}s exceeds {self.quality_thresholds['latency_maximum']:.1f}s threshold"
        
        # Performance quality checks
        assert max_latency < 2.0, f"Maximum latency {max_latency:.3f}s too high"
        assert min_latency > 0.01, f"Minimum latency {min_latency:.3f}s suspiciously low"
        
        logging.info(f"✅ Latency validation: {overall_avg_latency:.3f}s (±{latency_std:.3f}s)")
        logging.info(f"📊 Range: {min_latency:.3f}s - {max_latency:.3f}s on {len(latency_measurements)} measurements")
        
        self.test_results["latency_test"] = {
            "status": "PASSED",
            "average_seconds": overall_avg_latency,
            "std_dev_seconds": latency_std,
            "min_seconds": min_latency,
            "max_seconds": max_latency,
            "measurements_count": len(latency_measurements),
            "threshold_met": overall_avg_latency < self.quality_thresholds["latency_maximum"]
        }
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_efficiency_fp8(self, ai_processor, test_job_schema):
        """
        Memory efficiency testing validating FP8 quantization benefits.
        
        Validates memory usage stays within <4GB VRAM threshold under
        concurrent load while maintaining extraction quality.
        """
        
        # Baseline memory measurement
        baseline_stats = ai_processor.get_memory_stats()
        baseline_memory = baseline_stats["model_memory_gb"]
        
        logging.info(f"📊 Baseline memory: {baseline_memory:.1f}GB")
        
        # Concurrent load testing
        concurrent_tasks = []
        test_content_variations = [
            f"Test job posting {i}: Senior Software Engineer with Python, AWS, Docker, Kubernetes experience. "
            f"Remote position with competitive salary and comprehensive benefits package. "
            f"Requirements include {i}+ years of experience with cloud infrastructure."
            for i in range(8)
        ]
        
        # Execute concurrent requests
        start_time = time.monotonic()
        
        for i, content in enumerate(test_content_variations):
            task = ai_processor.extract_jobs(content, test_job_schema)
            concurrent_tasks.append(task)
        
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        end_time = time.monotonic()
        
        # Post-load memory measurement
        peak_stats = ai_processor.get_memory_stats()
        peak_memory = peak_stats.get("peak_memory_gb", baseline_memory)
        
        # Validate memory efficiency
        assert peak_memory <= self.quality_thresholds["memory_maximum_gb"], \
            f"Peak memory usage {peak_memory:.1f}GB exceeds {self.quality_thresholds['memory_maximum_gb']:.1f}GB threshold"
        
        # Validate extraction success under memory pressure
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append(f"Task {i}: {str(result)}")
            elif isinstance(result, dict):
                if result.get("_extraction_success", False):
                    successful_results.append(result)
                else:
                    failed_results.append(f"Task {i}: {result.get('error', 'Unknown error')}")
        
        success_rate = len(successful_results) / len(results)
        total_time = end_time - start_time
        
        assert success_rate >= 0.8, f"Success rate {success_rate:.1%} too low under memory load"
        
        # Memory efficiency metrics
        memory_efficiency = baseline_memory / 9.6  # Compared to full precision (~9.6GB)
        
        logging.info(f"✅ Memory validation: {peak_memory:.1f}GB peak, {success_rate:.1%} success rate")
        logging.info(f"📊 FP8 efficiency: {memory_efficiency:.1f}x memory reduction")
        
        self.test_results["memory_test"] = {
            "status": "PASSED", 
            "baseline_gb": baseline_memory,
            "peak_memory_gb": peak_memory,
            "memory_efficiency": f"{memory_efficiency:.1f}x reduction via FP8",
            "concurrent_success_rate": success_rate,
            "total_processing_time": total_time,
            "threshold_met": peak_memory <= self.quality_thresholds["memory_maximum_gb"]
        }
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput_performance(self, ai_processor, test_job_schema):
        """
        Throughput testing validating 45+ tokens/sec performance target.
        
        Measures token generation rate under sustained processing load
        with realistic job content scenarios.
        """
        
        # Generate realistic test jobs
        test_jobs = [
            f"Software Engineer {i}: Python, React, AWS experience required. "
            f"Full-time position with competitive salary and benefits. "
            f"Remote-friendly with flexible hours and professional development. "
            f"Team of {i+1} engineers working on scalable cloud solutions."
            for i in range(20)
        ]
        
        start_time = time.time()
        
        successful_extractions = 0
        total_tokens_generated = 0
        processing_times = []
        
        for job_content in test_jobs:
            job_start = time.monotonic()
            result = await ai_processor.extract_jobs(job_content, test_job_schema)
            job_end = time.monotonic()
            
            processing_times.append(job_end - job_start)
            
            if result.get("_extraction_success", False):
                successful_extractions += 1
                
                # Estimate tokens generated
                result_text = json.dumps(result)
                estimated_tokens = len(result_text) // 4  # Rough estimation
                total_tokens_generated += estimated_tokens
        
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        
        # Calculate throughput metrics
        tokens_per_second = total_tokens_generated / elapsed_seconds if elapsed_seconds > 0 else 0
        jobs_per_second = successful_extractions / elapsed_seconds if elapsed_seconds > 0 else 0
        avg_job_time = statistics.mean(processing_times) if processing_times else 0
        
        # Validate throughput threshold
        assert tokens_per_second >= self.quality_thresholds["throughput_minimum"], \
            f"Throughput {tokens_per_second:.1f} tokens/sec below {self.quality_thresholds['throughput_minimum']} threshold"
        
        # Additional throughput quality checks
        assert jobs_per_second >= 3, f"Jobs per second {jobs_per_second:.2f} too low"
        assert avg_job_time < 2.0, f"Average job processing time {avg_job_time:.2f}s too high"
        
        logging.info(f"✅ Throughput validation: {tokens_per_second:.1f} tokens/sec, {jobs_per_second:.2f} jobs/sec")
        logging.info(f"📊 Processed {successful_extractions}/{len(test_jobs)} jobs in {elapsed_seconds:.1f}s")
        
        self.test_results["throughput_test"] = {
            "status": "PASSED",
            "tokens_per_second": tokens_per_second,
            "jobs_per_second": jobs_per_second,
            "total_tokens": total_tokens_generated,
            "processing_time_seconds": elapsed_seconds,
            "success_rate": successful_extractions / len(test_jobs),
            "average_job_time": avg_job_time,
            "threshold_met": tokens_per_second >= self.quality_thresholds["throughput_minimum"]
        }
    
    # Reliability Testing Suite
    @pytest.mark.reliability
    @pytest.mark.asyncio 
    async def test_sustained_operation_reliability(self, ai_processor, test_job_schema):
        """
        Sustained operation reliability testing validating 95%+ uptime requirement.
        
        Simulates extended production usage with varied content and conditions
        to validate system stability and error handling resilience.
        """
        
        test_iterations = 100
        success_count = 0
        error_details = []
        latency_measurements = []
        memory_measurements = []
        
        # Generate diverse test content
        test_scenarios = [
            "Python Developer at StartupCorp with ML, FastAPI, PostgreSQL requirements",
            "Senior React Engineer - Remote position with TypeScript and Node.js",
            "DevOps Specialist: Kubernetes, Docker, AWS, Terraform experience needed", 
            "Data Scientist role requiring Python, TensorFlow, Pandas, SQL skills",
            "Full-stack Developer: React, Python, MongoDB, REST API experience",
            "Backend Engineer position with Go, microservices, database design",
            "Frontend Developer: Vue.js, JavaScript, CSS, responsive design",
            "Machine Learning Engineer with PyTorch, scikit-learn expertise"
        ]
        
        for i in range(test_iterations):
            # Vary content to test robustness
            scenario_idx = i % len(test_scenarios)
            test_content = f"Reliability test {i}: {test_scenarios[scenario_idx]}. "
            test_content += f"Position #{i} with {i%5 + 1} years experience requirement."
            
            try:
                start_time = time.monotonic()
                result = await ai_processor.extract_jobs(test_content, test_job_schema)
                end_time = time.monotonic()
                
                latency = end_time - start_time
                latency_measurements.append(latency)
                
                if result.get("_extraction_success", False):
                    success_count += 1
                    
                    # Validate extraction structure
                    if self._validate_extraction_structure(result):
                        # Structure validation passed
                        pass
                    else:
                        error_details.append(f"Iteration {i}: Invalid structure")
                        
                else:
                    error_details.append(f"Iteration {i}: {result.get('error', 'Unknown error')}")
                
                # Periodic memory monitoring
                if i % 20 == 0:
                    memory_stats = ai_processor.get_memory_stats()
                    memory_measurements.append({
                        "iteration": i,
                        "memory_gb": memory_stats.get("peak_memory_gb", 0),
                        "status": memory_stats.get("status", "unknown")
                    })
                    
            except Exception as e:
                error_details.append(f"Iteration {i}: Exception: {str(e)}")
                logging.warning(f"⚠️ Reliability test iteration {i} failed: {e}")
        
        # Calculate reliability metrics
        reliability = success_count / test_iterations
        avg_latency = statistics.mean(latency_measurements) if latency_measurements else 0
        latency_std = statistics.stdev(latency_measurements) if len(latency_measurements) > 1 else 0
        
        # Validate reliability threshold
        assert reliability >= self.quality_thresholds["reliability_minimum"], \
            f"Reliability {reliability:.1%} below {self.quality_thresholds['reliability_minimum']:.1%} threshold"
        
        # Additional reliability quality checks
        assert avg_latency < 2.0, f"Average latency {avg_latency:.3f}s too high under sustained load"
        assert len(error_details) <= 5, f"Too many errors ({len(error_details)}) in reliability test"
        
        # Memory stability check
        if memory_measurements:
            memory_values = [m["memory_gb"] for m in memory_measurements if m["memory_gb"] > 0]
            if memory_values:
                memory_trend = max(memory_values) - min(memory_values)
                assert memory_trend < 1.0, f"Memory usage increased by {memory_trend:.1f}GB during test"
        
        logging.info(f"✅ Reliability validation: {reliability:.1%} success rate ({success_count}/{test_iterations})")
        logging.info(f"📊 Latency: {avg_latency:.3f}s (±{latency_std:.3f}s), Errors: {len(error_details)}")
        
        if error_details[:3]:
            logging.debug(f"Sample errors: {error_details[:3]}")
        
        self.test_results["reliability_test"] = {
            "status": "PASSED",
            "success_rate": reliability,
            "successful_operations": success_count,
            "total_operations": test_iterations,
            "error_count": len(error_details),
            "average_latency": avg_latency,
            "latency_std_dev": latency_std,
            "memory_stability": "stable" if memory_measurements else "not_monitored",
            "threshold_met": reliability >= self.quality_thresholds["reliability_minimum"]
        }
    
    # Context and Edge Case Testing
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_context_length_handling(self, ai_processor, test_job_schema):
        """
        Context length handling test validating 8K optimization per ADR-034.
        
        Tests various content lengths to ensure proper handling of the 8K context
        window with graceful degradation for oversized content.
        """
        
        test_cases = [
            ("short", "Brief job posting." * 50),           # ~300 tokens
            ("medium", "Detailed requirements." * 300),      # ~3K tokens  
            ("optimal", "Comprehensive description." * 500), # ~6K tokens
            ("maximum", "Extensive job details." * 700),     # ~8-9K tokens (over limit)
            ("excessive", "Very long content." * 1000),      # ~12K tokens (well over limit)
        ]
        
        results_by_length = {}
        
        for case_name, content in test_cases:
            estimated_tokens = len(content) // 4
            
            try:
                start_time = time.monotonic()
                result = await ai_processor.extract_jobs(content, test_job_schema)
                end_time = time.monotonic()
                
                processing_time = end_time - start_time
                
                # Should handle all lengths gracefully (no exceptions)
                assert isinstance(result, dict), f"Failed to process {case_name} content"
                
                # Validate extraction success or proper error handling
                if result.get("_extraction_success", False):
                    # Successful extraction - validate content
                    assert any(key in result for key in ["title", "company", "description"]), \
                        f"Missing key fields in {case_name} extraction"
                    
                    # For longer content, validate truncation metadata
                    if estimated_tokens > 6000:
                        metadata = result.get("_extraction_metadata", {})
                        assert metadata.get("content_length_chars", 0) >= metadata.get("optimized_length_chars", 0), \
                            f"Content optimization metadata missing for {case_name}"
                
                results_by_length[case_name] = {
                    "estimated_tokens": estimated_tokens,
                    "processing_time": processing_time,
                    "success": result.get("_extraction_success", False),
                    "error": result.get("error", None),
                    "result_preview": str(result)[:200] + "..."
                }
                
                logging.debug(f"{case_name} ({estimated_tokens} tokens): "
                             f"{'✅' if result.get('_extraction_success') else '⚠️'} {processing_time:.3f}s")
                
            except Exception as e:
                pytest.fail(f"Exception processing {case_name} content: {e}")
        
        # Validate that shorter content has higher success rates
        short_success = results_by_length["short"]["success"]
        medium_success = results_by_length["medium"]["success"]
        
        assert short_success, "Short content should always succeed"
        assert medium_success, "Medium content should always succeed"
        
        # Validate that processing time scales reasonably
        short_time = results_by_length["short"]["processing_time"]
        excessive_time = results_by_length["excessive"]["processing_time"]
        
        assert excessive_time / short_time < 5, "Processing time should not scale dramatically with content length"
        
        logging.info(f"✅ Context length validation: Handled all content lengths gracefully")
        
        self.test_results["context_test"] = {
            "status": "PASSED",
            "results_by_length": results_by_length
        }
    
    # Quality Metrics and Reporting
    def test_generate_quality_report(self):
        """Generate comprehensive quality report from all test results."""
        
        if not self.test_results:
            pytest.skip("No test results available for quality report")
        
        # Calculate overall quality score
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        total_tests = len(self.test_results)
        overall_score = passed_tests / total_tests if total_tests > 0 else 0
        
        # Generate comprehensive report
        quality_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_quality_score": overall_score,
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "quality_thresholds": self.quality_thresholds,
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        # Validate overall quality meets standards
        assert overall_score >= 0.8, f"Overall quality score {overall_score:.1%} below 80% minimum"
        
        logging.info(f"✅ Quality Report: {overall_score:.1%} overall score ({passed_tests}/{total_tests} tests passed)")
        
        # Save report for CI/CD integration
        report_path = Path("test_results/quality_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        logging.info(f"📄 Quality report saved: {report_path}")
        
        return quality_report
    
    # Helper Methods
    def _calculate_extraction_accuracy(self, extracted: Dict, expected: Dict) -> float:
        """Calculate extraction accuracy score with weighted field importance."""
        
        required_fields = ["title", "company", "description"]
        optional_fields = ["location", "skills", "requirements", "salary_min", "salary_max", "employment_type"]
        
        correct_required = 0
        correct_optional = 0
        
        # Score required fields (higher weight)
        for field in required_fields:
            if field in extracted and field in expected:
                if self._fields_match(extracted[field], expected[field]):
                    correct_required += 1
        
        # Score optional fields (lower weight)  
        for field in optional_fields:
            if field in extracted and field in expected:
                if self._fields_match(extracted[field], expected[field]):
                    correct_optional += 1
        
        # Weighted accuracy calculation
        required_score = (correct_required / len(required_fields)) * 0.7
        optional_score = (correct_optional / len(optional_fields)) * 0.3
        
        return min(1.0, required_score + optional_score)
    
    def _fields_match(self, field1: Any, field2: Any, threshold: float = 0.7) -> bool:
        """Determine if extracted fields match expected values with fuzzy matching."""
        
        if field1 is None and field2 is None:
            return True
        if field1 is None or field2 is None:
            return False
            
        if isinstance(field1, str) and isinstance(field2, str):
            return self._strings_similar(field1, field2, threshold)
        elif isinstance(field1, list) and isinstance(field2, list):
            if not field1 or not field2:
                return len(field1) == len(field2) == 0
            set1 = set(str(x).lower().strip() for x in field1)
            set2 = set(str(x).lower().strip() for x in field2)
            overlap = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
            return overlap >= threshold
        elif isinstance(field1, (int, float)) and isinstance(field2, (int, float)):
            # Numeric fields with tolerance
            tolerance = max(abs(field1), abs(field2)) * 0.1  # 10% tolerance
            return abs(field1 - field2) <= tolerance
        else:
            return str(field1).lower().strip() == str(field2).lower().strip()
    
    def _strings_similar(self, s1: str, s2: str, threshold: float = 0.7) -> bool:
        """Calculate string similarity using Jaccard index on words."""
        
        if not s1 or not s2:
            return not s1 and not s2
        
        # Normalize strings
        s1_words = set(s1.lower().strip().split())
        s2_words = set(s2.lower().strip().split())
        
        if not s1_words or not s2_words:
            return s1.lower().strip() == s2.lower().strip()
        
        # Jaccard similarity
        intersection = len(s1_words & s2_words)
        union = len(s1_words | s2_words)
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def _validate_extraction_structure(self, extraction: Dict[str, Any]) -> bool:
        """Validate extraction has proper structure and required fields."""
        
        required_fields = ["title", "company", "description"]
        
        for field in required_fields:
            if field not in extraction:
                return False
            if not isinstance(extraction[field], str):
                return False
            if not extraction[field].strip():
                return False
        
        # Validate optional field types if present
        field_types = {
            "location": str,
            "skills": list,
            "requirements": list,
            "salary_min": (int, type(None)),
            "salary_max": (int, type(None)),
            "employment_type": (str, type(None)),
            "experience_level": (str, type(None))
        }
        
        for field, expected_type in field_types.items():
            if field in extraction:
                if not isinstance(extraction[field], expected_type):
                    return False
        
        return True
    
    def _generate_recommendations(self) -> List[str]:
        """Generate quality improvement recommendations based on test results."""
        
        recommendations = []
        
        for test_name, test_result in self.test_results.items():
            if test_result["status"] != "PASSED":
                continue
                
            # Analyze test-specific metrics for improvement opportunities
            if test_name == "accuracy_test":
                accuracy = test_result.get("average_accuracy", 0)
                if accuracy < 0.98:
                    recommendations.append(
                        f"Accuracy could be improved from {accuracy:.1%} - consider prompt tuning or fine-tuning"
                    )
                    
            elif test_name == "latency_test":
                avg_latency = test_result.get("average_seconds", 0)
                if avg_latency > 0.7:
                    recommendations.append(
                        f"Latency averaging {avg_latency:.3f}s - consider GPU optimization or batch processing"
                    )
                    
            elif test_name == "memory_test":
                peak_memory = test_result.get("peak_memory_gb", 0)
                if peak_memory > 3.5:
                    recommendations.append(
                        f"Peak memory usage {peak_memory:.1f}GB - monitor for memory leaks or optimization opportunities"
                    )
                    
            elif test_name == "reliability_test":
                success_rate = test_result.get("success_rate", 0)
                if success_rate < 0.98:
                    recommendations.append(
                        f"Reliability at {success_rate:.1%} - enhance error handling and retry mechanisms"
                    )
        
        if not recommendations:
            recommendations.append(
                "All quality metrics meet thresholds - maintain current performance and monitor for regression"
            )
        
        return recommendations

# Performance Monitoring and Continuous Testing
class ContinuousQualityMonitor:
    """Continuous quality monitoring for production deployment."""
    
    def __init__(self, ai_processor):
        self.ai_processor = ai_processor
        self.monitoring_active = False
        self.quality_metrics = {
            "hourly_accuracy": [],
            "hourly_latency": [], 
            "hourly_throughput": [],
            "error_count": 0,
            "total_requests": 0
        }
    
    async def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous quality monitoring."""
        
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Perform periodic quality checks
                await self._run_quality_check()
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logging.error(f"❌ Quality monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
    
    async def _run_quality_check(self):
        """Run periodic quality check with sample extraction."""
        
        # Sample test extraction
        test_content = "Quality Check: Python Developer position at TestCorp with AWS experience."
        test_schema = {
            "type": "object", 
            "properties": {
                "title": {"type": "string"},
                "company": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["title", "company", "description"]
        }
        
        start_time = time.monotonic()
        
        try:
            result = await self.ai_processor.extract_jobs(test_content, test_schema)
            end_time = time.monotonic()
            
            latency = end_time - start_time
            self.quality_metrics["total_requests"] += 1
            
            if result.get("_extraction_success", False):
                self.quality_metrics["hourly_latency"].append(latency)
                # Calculate approximate accuracy (simplified for monitoring)
                accuracy_estimate = 0.95 if "Python Developer" in result.get("title", "") else 0.8
                self.quality_metrics["hourly_accuracy"].append(accuracy_estimate)
            else:
                self.quality_metrics["error_count"] += 1
                
        except Exception as e:
            self.quality_metrics["error_count"] += 1
            logging.warning(f"⚠️ Quality check failed: {e}")
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get current quality monitoring summary."""
        
        summary = {
            "monitoring_active": self.monitoring_active,
            "total_requests": self.quality_metrics["total_requests"],
            "error_count": self.quality_metrics["error_count"],
            "error_rate": (
                self.quality_metrics["error_count"] / self.quality_metrics["total_requests"]
                if self.quality_metrics["total_requests"] > 0 else 0
            )
        }
        
        if self.quality_metrics["hourly_latency"]:
            summary["average_latency"] = statistics.mean(self.quality_metrics["hourly_latency"])
            
        if self.quality_metrics["hourly_accuracy"]:
            summary["average_accuracy"] = statistics.mean(self.quality_metrics["hourly_accuracy"])
        
        return summary

# Test Configuration and Utilities
def pytest_configure():
    """Configure pytest for AI quality testing."""
    
    # Add custom markers
    pytest.markers = [
        pytest.mark.integration("Integration testing with vLLM and FP8"),
        pytest.mark.accuracy("Accuracy validation against benchmarks"),
        pytest.mark.performance("Performance and latency testing"),
        pytest.mark.reliability("Reliability and stability testing"),
        pytest.mark.slow("Slow tests requiring extended runtime")
    ]

def pytest_collection_modifyitems(config, items):
    """Modify test collection for CI/CD optimization."""
    
    # Skip slow tests in fast CI mode
    if config.getoption("--fast"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests in fast mode")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

# Command Line Interface for Testing
def main():
    """Command line interface for AI quality testing."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Quality Testing - Implementation of ADR-035"
    )
    parser.add_argument(
        "command",
        choices=["run", "report", "monitor", "validate"],
        help="Testing command to execute"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Run fast test subset"
    )
    parser.add_argument(
        "--markers",
        help="Pytest markers to run (e.g., 'integration or performance')"
    )
    parser.add_argument(
        "--output",
        help="Output file for test report"
    )
    
    args = parser.parse_args()
    
    if args.command == "run":
        # Run pytest with specified options
        pytest_args = ["-v", "--tb=short"]
        
        if args.fast:
            pytest_args.extend(["-m", "not slow"])
        
        if args.markers:
            pytest_args.extend(["-m", args.markers])
        
        if args.output:
            pytest_args.extend(["--junitxml", args.output])
        
        exit_code = pytest.main(pytest_args)
        sys.exit(exit_code)
        
    elif args.command == "report":
        # Generate quality report
        print("📊 Generating quality report...")
        # Implementation would load test results and generate report
        
    elif args.command == "monitor":
        # Start continuous monitoring
        print("🔄 Starting continuous quality monitoring...")
        # Implementation would start monitoring service
        
    elif args.command == "validate":
        # Validate test environment
        print("🔍 Validating test environment...")
        # Implementation would check dependencies and configuration

if __name__ == "__main__":
    main()