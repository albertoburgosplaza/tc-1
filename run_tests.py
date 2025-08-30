#!/usr/bin/env python3
"""
Test execution script for RAG Chatbot System
Runs all test suites and generates comprehensive reports
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse


class TestRunner:
    """Comprehensive test runner for the RAG system"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def run_command(self, command: List[str], description: str, 
                   timeout: int = 300) -> Dict[str, Any]:
        """Run a command and capture results"""
        print(f"🏃 {description}...")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            success = result.returncode == 0
            
            test_result = {
                'command': ' '.join(command),
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time,
                'timestamp': time.time()
            }
            
            status = "✅" if success else "❌"
            print(f"{status} {description} - {execution_time:.1f}s")
            
            if not success:
                print(f"   Error: {result.stderr[:200]}...")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} - TIMEOUT after {timeout}s")
            return {
                'command': ' '.join(command),
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'execution_time': timeout,
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"💥 {description} - ERROR: {str(e)}")
            return {
                'command': ' '.join(command),
                'success': False,
                'returncode': -2,
                'stdout': '',
                'stderr': str(e),
                'execution_time': 0,
                'timestamp': time.time()
            }
    
    def run_unit_tests(self) -> bool:
        """Run unit tests"""
        command = ['python3', '-m', 'pytest', 'tests/unit/', '-v', '--tb=short']
        result = self.run_command(command, "Unit Tests")
        self.results['unit_tests'] = result
        return result['success']
    
    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        command = ['python3', '-m', 'pytest', 'tests/integration/', '-v', '--tb=short', '-m', 'integration']
        result = self.run_command(command, "Integration Tests")
        self.results['integration_tests'] = result
        return result['success']
    
    def run_e2e_tests(self) -> bool:
        """Run end-to-end tests"""
        command = ['python3', '-m', 'pytest', 'tests/e2e/', '-v', '--tb=short', '-m', 'e2e']
        result = self.run_command(command, "End-to-End Tests", timeout=600)
        self.results['e2e_tests'] = result
        return result['success']
    
    def run_evaluation_tests(self) -> bool:
        """Run RAG quality evaluation tests"""
        command = ['python3', '-m', 'pytest', 'tests/evaluation/', '-v', '--tb=short', '-m', 'evaluation']
        result = self.run_command(command, "RAG Quality Evaluation", timeout=300)
        self.results['evaluation_tests'] = result
        return result['success']
    
    def run_acceptance_criteria(self) -> bool:
        """Run acceptance criteria validation"""
        command = ['python3', 'validate_acceptance_criteria.py']
        result = self.run_command(command, "Acceptance Criteria Validation")
        self.results['acceptance_criteria'] = result
        return result['success']
    
    def run_docker_tests(self) -> bool:
        """Run Docker-based tests"""
        command = ['docker-compose', '-f', 'docker-compose.test.yml', 'config']
        result = self.run_command(command, "Docker Compose Configuration Test", timeout=60)
        self.results['docker_config'] = result
        return result['success']
    
    def run_linting(self) -> bool:
        """Run code linting (if available)"""
        # Check if pytest is available first
        try:
            subprocess.run(['python3', '-c', 'import pytest'], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("⚠️ Pytest not available, skipping linting")
            self.results['linting'] = {
                'command': 'python3 -c "import pytest"',
                'success': False,
                'returncode': 1,
                'stdout': '',
                'stderr': 'pytest not installed',
                'execution_time': 0,
                'timestamp': time.time()
            }
            return False
        
        # Run basic structure validation
        command = ['python3', '-m', 'py_compile'] + [
            'app.py', 'ingest.py', 'pyexec_service.py', 
            'validate_acceptance_criteria.py'
        ]
        result = self.run_command(command, "Code Compilation Check", timeout=30)
        self.results['linting'] = result
        return result['success']
    
    def generate_coverage_report(self) -> bool:
        """Generate test coverage report"""
        if not Path('tests').exists():
            print("⚠️ Tests directory not found")
            return False
        
        command = ['python3', '-m', 'pytest', '--cov=.', '--cov-report=html', 
                  '--cov-report=term', 'tests/', '--tb=no', '-q']
        result = self.run_command(command, "Coverage Report Generation", timeout=180)
        self.results['coverage'] = result
        return result['success']
    
    def run_all_tests(self, include_slow: bool = False, 
                     include_docker: bool = False) -> Dict[str, Any]:
        """Run all test suites"""
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        test_suites = [
            ("Code Validation", self.run_linting),
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Acceptance Criteria", self.run_acceptance_criteria),
        ]
        
        if include_slow:
            test_suites.extend([
                ("RAG Evaluation", self.run_evaluation_tests),
                ("Coverage Report", self.generate_coverage_report),
            ])
        
        if include_docker:
            test_suites.extend([
                ("Docker Configuration", self.run_docker_tests),
                ("End-to-End Tests", self.run_e2e_tests),
            ])
        
        passed_suites = 0
        total_suites = len(test_suites)
        
        for suite_name, test_method in test_suites:
            print(f"\n📋 Running {suite_name}...")
            try:
                success = test_method()
                if success:
                    passed_suites += 1
                    print(f"✅ {suite_name} completed successfully")
                else:
                    print(f"❌ {suite_name} failed")
            except Exception as e:
                print(f"💥 {suite_name} crashed: {str(e)}")
        
        # Generate summary
        total_execution_time = time.time() - self.start_time
        
        summary = {
            'test_run_timestamp': time.time(),
            'total_execution_time': total_execution_time,
            'total_suites': total_suites,
            'passed_suites': passed_suites,
            'failed_suites': total_suites - passed_suites,
            'success_rate': passed_suites / total_suites if total_suites > 0 else 0,
            'all_passed': passed_suites == total_suites,
            'suite_results': self.results,
            'configuration': {
                'include_slow': include_slow,
                'include_docker': include_docker
            }
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total Execution Time: {total_execution_time:.1f}s")
        print(f"Test Suites Run: {total_suites}")
        print(f"Passed: {passed_suites} ✅")
        print(f"Failed: {total_suites - passed_suites} ❌")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        
        if summary['all_passed']:
            print("\n🎉 ALL TEST SUITES PASSED!")
            print("The system is ready for deployment.")
        else:
            print("\n⚠️ SOME TEST SUITES FAILED")
            
            failed_suites = [name for name, result in self.results.items() 
                           if not result.get('success', False)]
            print(f"Failed suites: {', '.join(failed_suites)}")
        
        return summary
    
    def save_report(self, filename: str = "test_execution_report.json"):
        """Save comprehensive test report"""
        report_data = {
            'test_execution_summary': {
                'timestamp': time.time(),
                'total_time': time.time() - self.start_time,
                'results': self.results
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Comprehensive report saved to {filename}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run RAG System Test Suite")
    parser.add_argument('--slow', action='store_true', 
                       help='Include slow tests (evaluation, coverage)')
    parser.add_argument('--docker', action='store_true',
                       help='Include Docker-based tests (requires Docker)')
    parser.add_argument('--suite', choices=['unit', 'integration', 'e2e', 'evaluation', 'acceptance'],
                       help='Run specific test suite only')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.suite:
            # Run specific suite
            suite_methods = {
                'unit': runner.run_unit_tests,
                'integration': runner.run_integration_tests,
                'e2e': runner.run_e2e_tests,
                'evaluation': runner.run_evaluation_tests,
                'acceptance': runner.run_acceptance_criteria
            }
            
            if args.suite in suite_methods:
                print(f"🎯 Running {args.suite} tests only...")
                success = suite_methods[args.suite]()
                sys.exit(0 if success else 1)
            else:
                print(f"❌ Unknown test suite: {args.suite}")
                sys.exit(2)
        else:
            # Run all tests
            summary = runner.run_all_tests(
                include_slow=args.slow,
                include_docker=args.docker
            )
            runner.save_report()
            
            sys.exit(0 if summary['all_passed'] else 1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        sys.exit(3)


if __name__ == "__main__":
    main()