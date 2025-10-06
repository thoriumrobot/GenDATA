#!/usr/bin/env python3
"""
Semantic Augmentation Test Suite

Comprehensive test suite to verify that each semantic augmentation method
behaves as intended, preserving semantics while applying transformations.
"""

import os
import re
import json
import logging
import unittest
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import tempfile

# Import augmentation systems
from enhanced_semantic_augment_slices import EnhancedSemanticTransformer
from simple_code_semantic_augment_slices import SimpleCodeSemanticTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Individual test case for semantic augmentation."""
    name: str
    description: str
    input_code: str
    expected_patterns: List[str]
    should_transform: bool
    semantic_equivalence_checks: List[str]
    system_type: str  # 'enhanced' or 'simple'

@dataclass
class TestResult:
    """Result of a single test case."""
    test_name: str
    passed: bool
    error_message: Optional[str]
    transformation_applied: bool
    pattern_matches: List[str]
    semantic_checks_passed: List[bool]

class SemanticAugmentationTestSuite(unittest.TestCase):
    """Test suite for semantic augmentation methods."""
    
    def setUp(self):
        """Set up test environment."""
        self.enhanced_transformer = EnhancedSemanticTransformer(seed=42)
        self.simple_transformer = SimpleCodeSemanticTransformer(seed=42)
        
        # Load test cases
        self.enhanced_test_cases = self._load_enhanced_test_cases()
        self.simple_test_cases = self._load_simple_test_cases()
    
    def _load_enhanced_test_cases(self) -> List[TestCase]:
        """Load test cases for enhanced semantic augmentations."""
        return [
            # Loop conversions
            TestCase(
                name="for_to_while_conversion",
                description="Convert for loop to while loop",
                input_code="""
public class TestClass {
    public void testMethod() {
        for (int i = 0; i < 10; i++) {
            System.out.println(i);
        }
    }
}""",
                expected_patterns=[
                    r'int\s+i\s*=\s*0\s*;',
                    r'while\s*\(\s*i\s*<\s*10\s*\)',
                    r'i\+\+\s*;'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Loop executes same number of times",
                    "Variable i has same values",
                    "Same output produced"
                ],
                system_type="enhanced"
            ),
            
            # Guard reversals
            TestCase(
                name="guard_reversal",
                description="Reverse if-else conditions",
                input_code="""
public class TestClass {
    public String testMethod(int x) {
        if (x > 0) {
            return "positive";
        } else {
            return "negative";
        }
    }
}""",
                expected_patterns=[
                    r'if\s*\(\s*0\s*<\s*x\s*\)',
                    r'if\s*\(\s*x\s*<=\s*0\s*\)'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same return values for all inputs",
                    "Logical equivalence maintained"
                ],
                system_type="enhanced"
            ),
            
            # Mathematical expressions
            TestCase(
                name="mathematical_commutativity",
                description="Apply mathematical commutativity",
                input_code="""
public class TestClass {
    public int calculate(int a, int b) {
        return a + b + 0;
    }
}""",
                expected_patterns=[
                    r'a\s*\+\s*b',
                    r'b\s*\+\s*a',
                    r'0\s*\+\s*a',
                    r'a\s*\+\s*0'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Mathematical result unchanged",
                    "Commutative property preserved"
                ],
                system_type="enhanced"
            ),
            
            # De Morgan's laws
            TestCase(
                name="demorgans_laws",
                description="Apply De Morgan's laws",
                input_code="""
public class TestClass {
    public boolean testMethod(int a, int b) {
        return !(a > 0 && b > 0);
    }
}""",
                expected_patterns=[
                    r'a\s*<=\s*0\s*\|\|\s*b\s*<=\s*0',
                    r'b\s*<=\s*0\s*\|\|\s*a\s*<=\s*0'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Logical equivalence maintained",
                    "Truth table preserved"
                ],
                system_type="enhanced"
            ),
            
            # Ternary operators
            TestCase(
                name="ternary_to_if_else",
                description="Convert ternary operator to if-else",
                input_code="""
public class TestClass {
    public String testMethod(int x) {
        return x > 0 ? "positive" : "negative";
    }
}""",
                expected_patterns=[
                    r'if\s*\(\s*x\s*>\s*0\s*\)\s*\{',
                    r'return\s+"positive"\s*;',
                    r'\}\s*else\s*\{',
                    r'return\s+"negative"\s*;'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same conditional logic",
                    "Same return values"
                ],
                system_type="enhanced"
            ),
            
            # Switch statements
            TestCase(
                name="switch_to_if_else",
                description="Convert switch to if-else chain",
                input_code="""
public class TestClass {
    public String testMethod(int x) {
        switch (x) {
            case 1: return "one";
            case 2: return "two";
            default: return "other";
        }
    }
}""",
                expected_patterns=[
                    r'if\s*\(\s*x\s*==\s*1\s*\)',
                    r'if\s*\(\s*x\s*==\s*2\s*\)'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same case matching logic",
                    "Same return values"
                ],
                system_type="enhanced"
            ),
            
            # Variable operations
            TestCase(
                name="variable_inlining",
                description="Inline variable assignments",
                input_code="""
public class TestClass {
    public int testMethod() {
        int temp = 5;
        return temp + 10;
    }
}""",
                expected_patterns=[
                    r'return\s+5\s*\+\s*10',
                    r'return\s+10\s*\+\s*5'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same computation result",
                    "Variable elimination correct"
                ],
                system_type="enhanced"
            ),
            
            # Method extraction
            TestCase(
                name="method_extraction",
                description="Extract complex expression into method",
                input_code="""
public class TestClass {
    public int testMethod() {
        return (5 * 2) + (10 / 2);
    }
}""",
                expected_patterns=[
                    r'private.*int.*computeResult',
                    r'return\s+computeResult\(\s*\)',
                    r'return\s+\(5\s*\*\s*2\)\s*\+\s*\(10\s*/\s*2\)'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same mathematical result",
                    "Method extraction preserves semantics"
                ],
                system_type="enhanced"
            )
        ]
    
    def _load_simple_test_cases(self) -> List[TestCase]:
        """Load test cases for simple semantic augmentations."""
        return [
            # Simple method calls
            TestCase(
                name="simple_method_call_variations",
                description="Add parentheses and spacing to method calls",
                input_code="""
public class TestClass {
    public void testMethod() {
        String result = getValue();
        System.out.println(result);
    }
    
    public String getValue() {
        return "test";
    }
}""",
                expected_patterns=[
                    r'\(\s*getValue\s*\)\s*\(\s*\)',
                    r'System\.out\.println\s*\('
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same method invocation",
                    "Same execution flow"
                ],
                system_type="simple"
            ),
            
            # Simple assignments
            TestCase(
                name="simple_assignment_variations",
                description="Modify assignment spacing and format",
                input_code="""
public class TestClass {
    public void testMethod() {
        int x = 5;
        String s = "hello";
    }
}""",
                expected_patterns=[
                    r'int\s+x=5',
                    r'String\s+s\s*=\s*"hello"\s*;'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same variable values",
                    "Same assignment semantics"
                ],
                system_type="simple"
            ),
            
            # Simple conditionals
            TestCase(
                name="simple_conditional_variations",
                description="Modify simple conditional expressions",
                input_code="""
public class TestClass {
    public void testMethod(int x) {
        if (x > 0) {
            System.out.println("positive");
        }
    }
}""",
                expected_patterns=[
                    r'if\s*\(\s*0\s*<\s*x\s*\)',
                    r'if\s*\(\s*!.*!.*x.*>.*0.*\)'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same conditional logic",
                    "Same execution path"
                ],
                system_type="simple"
            ),
            
            # Simple array access
            TestCase(
                name="simple_array_access_variations",
                description="Modify array access patterns",
                input_code="""
public class TestClass {
    public void testMethod(int[] arr) {
        int x = arr[0];
        int y = arr[1];
    }
}""",
                expected_patterns=[
                    r'arr\s*\[\s*0\s*\+\s*0\s*\]',
                    r'arr\s*\[\s*1\s*\+\s*0\s*\]'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same array access",
                    "Same array indices"
                ],
                system_type="simple"
            ),
            
            # Simple return statements
            TestCase(
                name="simple_return_variations",
                description="Modify return statement format",
                input_code="""
public class TestClass {
    public int testMethod() {
        return 42;
    }
}""",
                expected_patterns=[
                    r'return\s*\(\s*42\s*\)',
                    r'return\s+0\s*\+\s*42'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same return value",
                    "Same method behavior"
                ],
                system_type="simple"
            ),
            
            # Simple variable declarations
            TestCase(
                name="simple_variable_declaration_variations",
                description="Modify variable declaration format",
                input_code="""
public class TestClass {
    public void testMethod() {
        int x = 5;
        String s = "test";
    }
}""",
                expected_patterns=[
                    r'final\s+int\s+x\s*=\s*5',
                    r'final\s+String\s+s\s*=\s*"test"'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same variable semantics",
                    "Same variable values"
                ],
                system_type="simple"
            ),
            
            # Simple constructor calls
            TestCase(
                name="simple_constructor_variations",
                description="Modify constructor call format",
                input_code="""
public class TestClass {
    public void testMethod() {
        String s = new String("test");
        int[] arr = new int[10];
    }
}""",
                expected_patterns=[
                    r'\(\s*new\s+String\s*\(',
                    r'new\s+int\s*\[\s*0\s*\+\s*10\s*\]'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same object creation",
                    "Same constructor semantics"
                ],
                system_type="simple"
            ),
            
            # Simple field access
            TestCase(
                name="simple_field_access_variations",
                description="Modify field access format",
                input_code="""
public class TestClass {
    public void testMethod(String s) {
        int len = s.length();
        char c = s.charAt(0);
    }
}""",
                expected_patterns=[
                    r'\(\s*s\s*\)\s*\.\s*length',
                    r'\(\s*s\s*\)\s*\.\s*charAt'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same field access",
                    "Same method invocation"
                ],
                system_type="simple"
            ),
            
            # Simple string operations
            TestCase(
                name="simple_string_variations",
                description="Modify string literal format",
                input_code="""
public class TestClass {
    public void testMethod() {
        String s = "hello";
        String t = s + " world";
    }
}""",
                expected_patterns=[
                    r'\(\s*"hello"\s*\)',
                    r'\(\s*" world"\s*\)'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same string values",
                    "Same string concatenation"
                ],
                system_type="simple"
            ),
            
            # Simple numeric operations
            TestCase(
                name="simple_numeric_variations",
                description="Apply numeric identity operations",
                input_code="""
public class TestClass {
    public int testMethod(int x) {
        return x + 0;
    }
}""",
                expected_patterns=[
                    r'return\s+x\s*;',
                    r'return\s+0\s*\+\s*x'
                ],
                should_transform=True,
                semantic_equivalence_checks=[
                    "Same numeric result",
                    "Mathematical identity preserved"
                ],
                system_type="simple"
            )
        ]
    
    def test_enhanced_augmentations(self):
        """Test all enhanced semantic augmentation methods."""
        for test_case in self.enhanced_test_cases:
            with self.subTest(test_case=test_case.name):
                result = self._run_test_case(test_case, self.enhanced_transformer)
                
                # Check if transformation was applied
                if test_case.should_transform:
                    self.assertTrue(result.transformation_applied, 
                                  f"Expected transformation for {test_case.name}")
                
                # Check pattern matches
                self.assertGreater(len(result.pattern_matches), 0,
                                 f"No expected patterns matched for {test_case.name}")
                
                # Check semantic equivalence
                semantic_passed = all(result.semantic_checks_passed)
                self.assertTrue(semantic_passed,
                              f"Semantic equivalence failed for {test_case.name}")
    
    def test_simple_augmentations(self):
        """Test all simple semantic augmentation methods."""
        for test_case in self.simple_test_cases:
            with self.subTest(test_case=test_case.name):
                result = self._run_test_case(test_case, self.simple_transformer)
                
                # Check if transformation was applied
                if test_case.should_transform:
                    self.assertTrue(result.transformation_applied,
                                  f"Expected transformation for {test_case.name}")
                
                # Check pattern matches
                self.assertGreater(len(result.pattern_matches), 0,
                                 f"No expected patterns matched for {test_case.name}")
                
                # Check semantic equivalence
                semantic_passed = all(result.semantic_checks_passed)
                self.assertTrue(semantic_passed,
                              f"Semantic equivalence failed for {test_case.name}")
    
    def _run_test_case(self, test_case: TestCase, transformer) -> TestResult:
        """Run a single test case and return results."""
        try:
            # Get transformation method
            method_name = f'_transform_{test_case.name.split("_")[0]}'
            if test_case.system_type == "simple":
                method_name = f'_transform_{test_case.name}'
            
            # Find the actual method name
            method_name = self._find_transformation_method(test_case, transformer)
            
            if method_name is None:
                return TestResult(
                    test_name=test_case.name,
                    passed=False,
                    error_message=f"Transformation method not found for {test_case.name}",
                    transformation_applied=False,
                    pattern_matches=[],
                    semantic_checks_passed=[]
                )
            
            transform_method = getattr(transformer, method_name)
            
            # Apply transformation
            original_code = test_case.input_code
            transformed_code = transform_method(original_code)
            
            # Check if transformation was applied
            transformation_applied = transformed_code != original_code
            
            # Check pattern matches
            pattern_matches = []
            for pattern in test_case.expected_patterns:
                if re.search(pattern, transformed_code, re.MULTILINE | re.DOTALL):
                    pattern_matches.append(pattern)
            
            # Check semantic equivalence (simplified checks)
            semantic_checks_passed = self._check_semantic_equivalence(
                test_case, original_code, transformed_code
            )
            
            passed = (transformation_applied == test_case.should_transform and 
                     len(pattern_matches) > 0 and 
                     all(semantic_checks_passed))
            
            return TestResult(
                test_name=test_case.name,
                passed=passed,
                error_message=None,
                transformation_applied=transformation_applied,
                pattern_matches=pattern_matches,
                semantic_checks_passed=semantic_checks_passed
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_case.name,
                passed=False,
                error_message=str(e),
                transformation_applied=False,
                pattern_matches=[],
                semantic_checks_passed=[]
            )
    
    def _find_transformation_method(self, test_case: TestCase, transformer) -> Optional[str]:
        """Find the correct transformation method name for a test case."""
        # Map test case names to method names
        method_mapping = {
            'for_to_while_conversion': '_transform_loops',
            'guard_reversal': '_transform_guards',
            'mathematical_commutativity': '_transform_mathematical_expressions',
            'demorgans_laws': '_transform_logical_expressions',
            'ternary_to_if_else': '_transform_ternary_operators',
            'switch_to_if_else': '_transform_switch_statements',
            'variable_inlining': '_transform_variable_operations',
            'method_extraction': '_transform_method_extraction',
            'simple_method_call_variations': '_transform_simple_method_calls',
            'simple_assignment_variations': '_transform_simple_assignments',
            'simple_conditional_variations': '_transform_simple_conditionals',
            'simple_array_access_variations': '_transform_simple_array_access',
            'simple_return_variations': '_transform_simple_return_statements',
            'simple_variable_declaration_variations': '_transform_simple_variable_declarations',
            'simple_constructor_variations': '_transform_simple_constructor_calls',
            'simple_field_access_variations': '_transform_simple_field_access',
            'simple_string_variations': '_transform_simple_string_operations',
            'simple_numeric_variations': '_transform_simple_numeric_operations'
        }
        
        method_name = method_mapping.get(test_case.name)
        if method_name and hasattr(transformer, method_name):
            return method_name
        
        return None
    
    def _check_semantic_equivalence(self, test_case: TestCase, original_code: str, 
                                  transformed_code: str) -> List[bool]:
        """Check semantic equivalence between original and transformed code."""
        checks = []
        
        # Basic checks based on test case type
        if "loop" in test_case.name:
            # Check that loop structure is preserved
            original_has_loop = bool(re.search(r'for\s*\(|while\s*\(', original_code))
            transformed_has_loop = bool(re.search(r'for\s*\(|while\s*\(', transformed_code))
            checks.append(original_has_loop or transformed_has_loop)
        
        if "conditional" in test_case.name or "guard" in test_case.name:
            # Check that conditional logic is preserved
            original_has_conditional = bool(re.search(r'if\s*\(', original_code))
            transformed_has_conditional = bool(re.search(r'if\s*\(', transformed_code))
            checks.append(original_has_conditional and transformed_has_conditional)
        
        if "mathematical" in test_case.name or "numeric" in test_case.name:
            # Check that mathematical operations are preserved
            original_has_math = bool(re.search(r'[\+\-\*/]', original_code))
            transformed_has_math = bool(re.search(r'[\+\-\*/]', transformed_code))
            checks.append(original_has_math and transformed_has_math)
        
        if "return" in test_case.name:
            # Check that return statements are preserved
            original_returns = len(re.findall(r'return\s+', original_code))
            transformed_returns = len(re.findall(r'return\s+', transformed_code))
            checks.append(original_returns == transformed_returns)
        
        # Default check: both codes should be compilable Java
        checks.append(self._is_valid_java(original_code))
        checks.append(self._is_valid_java(transformed_code))
        
        return checks
    
    def _is_valid_java(self, code: str) -> bool:
        """Basic check if code looks like valid Java."""
        # Check for basic Java structure
        has_class = bool(re.search(r'class\s+\w+', code))
        has_braces = code.count('{') == code.count('}')
        has_semicolons = code.count(';') > 0
        
        return has_class and has_braces and has_semicolons


class SemanticAugmentationTestRunner:
    """Test runner for semantic augmentation test suite."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all semantic augmentation tests."""
        logger.info("Running semantic augmentation test suite...")
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add enhanced augmentation tests
        enhanced_loader = unittest.TestLoader()
        enhanced_tests = enhanced_loader.loadTestsFromName('test_enhanced_augmentations')
        test_suite.addTests(enhanced_tests)
        
        # Add simple augmentation tests
        simple_loader = unittest.TestLoader()
        simple_tests = simple_loader.loadTestsFromName('test_simple_augmentations')
        test_suite.addTests(simple_tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        # Collect results
        test_results = {
            'total_tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun,
            'failure_details': [str(failure) for failure in result.failures],
            'error_details': [str(error) for error in result.errors]
        }
        
        # Save results
        self._save_test_results(test_results)
        
        logger.info(f"Test suite complete: {test_results['total_tests']} tests, "
                   f"{test_results['success_rate']:.2%} success rate")
        
        return test_results
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to JSON file."""
        output_file = os.path.join(self.output_dir, 'semantic_augmentation_test_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {output_file}")


def main():
    """Main function to run semantic augmentation test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run semantic augmentation test suite')
    parser.add_argument('--output_dir', 
                       default='/home/ubuntu/GenDATA/test_results/',
                       help='Output directory for test results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose test output')
    
    args = parser.parse_args()
    
    # Create test runner
    test_runner = SemanticAugmentationTestRunner(args.output_dir)
    
    # Run tests
    results = test_runner.run_all_tests()
    
    # Print summary
    print("\n" + "="*50)
    print("SEMANTIC AUGMENTATION TEST SUITE SUMMARY")
    print("="*50)
    print(f"Total tests: {results['total_tests']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    
    if results['failures'] > 0:
        print(f"\nFailures:")
        for failure in results['failure_details']:
            print(f"  - {failure}")
    
    if results['errors'] > 0:
        print(f"\nErrors:")
        for error in results['error_details']:
            print(f"  - {error}")


if __name__ == '__main__':
    main()
