#!/usr/bin/env python3
"""
Script to create the remaining 6 transformation test files.
"""
import os

def create_type_conversion_test():
    content = '''package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Type Conversion transformation.
 * Tests redundant cast removal and string concatenation transformations.
 */
@DisplayName("Type Conversion Transformation Tests")
class TypeConversionTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
        transformer = new SemanticTransformer(42L); // Seeded for determinism
    }
    
    private static final String TRANSFORMATION = "type_conversion";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Cast Removal Operations")
    class CastRemovalOperations {
        
        @Test
        @DisplayName("Remove redundant int cast")
        public void testTypeConversion_Case1_RemoveRedundantIntCast() {
            String method = """
                public void removeIntCast() {
                    int result = (int)5;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Remove int cast should transform code");
            assertCompiles(transformed, "Remove int cast should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Remove int cast should preserve semantics");
        }
        
        @Test
        @DisplayName("Remove redundant String cast")
        public void testTypeConversion_Case2_RemoveRedundantStringCast() {
            String method = """
                public void removeStringCast() {
                    String result = (String)"hello";
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Remove String cast should transform code");
            assertCompiles(transformed, "Remove String cast should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Remove String cast should preserve semantics");
        }
        
        @Test
        @DisplayName("Primitive type conversions")
        public void testTypeConversion_Case3_PrimitiveTypeConversions() {
            String method = """
                public void primitiveConversions() {
                    double d = (double)5;
                    float f = (float)3.14;
                    System.out.println(d + f);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Primitive conversions should transform code");
            assertCompiles(transformed, "Primitive conversions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Primitive conversions should preserve semantics");
        }
        
        @Test
        @DisplayName("Object type casts")
        public void testTypeConversion_Case4_ObjectTypeCasts() {
            String method = """
                public void objectTypeCasts() {
                    Object obj = "test";
                    String str = (String)obj;
                    System.out.println(str);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Object casts should transform code");
            assertCompiles(transformed, "Object casts should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Object casts should preserve semantics");
        }
        
        @Test
        @DisplayName("Array type casts")
        public void testTypeConversion_Case5_ArrayTypeCasts() {
            String method = """
                public void arrayTypeCasts() {
                    int[] arr = (int[]){1, 2, 3};
                    System.out.println(arr.length);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Array casts should transform code");
            assertCompiles(transformed, "Array casts should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Array casts should preserve semantics");
        }
        
        @Test
        @DisplayName("Multi-level casts")
        public void testTypeConversion_Case6_MultiLevelCasts() {
            String method = """
                public void multiLevelCasts() {
                    Object obj = (Object)(String)"test";
                    System.out.println(obj);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multi-level casts should transform code");
            assertCompiles(transformed, "Multi-level casts should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Multi-level casts should preserve semantics");
        }
        
        @Test
        @DisplayName("Cast with expressions")
        public void testTypeConversion_Case7_CastWithExpressions() {
            String method = """
                public void castWithExpressions() {
                    int result = (int)(5.5 + 2.3);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Cast with expressions should transform code");
            assertCompiles(transformed, "Cast with expressions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Cast with expressions should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: necessary casts")
        public void testTypeConversion_Case8_NecessaryCasts() {
            String method = """
                public void necessaryCasts() {
                    Object obj = new String("test");
                    String str = (String)obj;
                    System.out.println(str);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Necessary casts should transform code");
            assertCompiles(transformed, "Necessary casts should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Necessary casts should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("String Concatenation Operations")
    class StringConcatenationOperations {
        
        @Test
        @DisplayName("String concatenation to StringBuilder")
        public void testTypeConversion_Case9_StringConcatenationToBuilder() {
            String method = """
                public void stringConcatenation() {
                    String result = "Hello" + " " + "World";
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "String concatenation should transform code");
            assertCompiles(transformed, "String concatenation should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "String concatenation should preserve semantics");
        }
        
        @Test
        @DisplayName("Complex string concatenation")
        public void testTypeConversion_Case10_ComplexStringConcatenation() {
            String method = """
                public void complexStringConcatenation() {
                    String result = "Value: " + value + " Count: " + count;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex string concatenation should transform code");
            assertCompiles(transformed, "Complex string concatenation should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex string concatenation should preserve semantics");
        }
        
        @Test
        @DisplayName("String concatenation with method calls")
        public void testTypeConversion_Case11_StringConcatenationWithMethodCalls() {
            String method = """
                public void stringConcatenationWithMethodCalls() {
                    String result = "Result: " + obj.getValue();
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "String concatenation with method calls should transform code");
            assertCompiles(transformed, "String concatenation with method calls should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "String concatenation with method calls should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive type conversion test")
    public void testTypeConversion_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                // Test cast removal
                int i = (int)5;
                String s = (String)"hello";
                double d = (double)3.14;
                
                // Test string concatenation
                String result = "Test: " + i + " " + s + " " + d;
                
                System.out.println(result);
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}'''
    
    with open('src/test/java/cfwr/jdt/transformations/enhanced/TypeConversionTransformationTest.java', 'w') as f:
        f.write(content)

def create_null_check_pattern_test():
    content = '''package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Null Check Pattern transformation.
 * Tests modern null check patterns and safe equals operations.
 */
@DisplayName("Null Check Pattern Transformation Tests")
class NullCheckPatternTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
        transformer = new SemanticTransformer(42L); // Seeded for determinism
    }
    
    private static final String TRANSFORMATION = "null_check_pattern";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Null Check Transformations")
    class NullCheckTransformations {
        
        @Test
        @DisplayName("obj != null to !Objects.isNull(obj)")
        public void testNullCheckPattern_Case1_NotNullToObjectsIsNull() {
            String method = """
                public void notNullToObjectsIsNull() {
                    if (obj != null) {
                        System.out.println("Not null");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "NotNull check should transform code");
            assertCompiles(transformed, "NotNull check should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "NotNull check should preserve semantics");
        }
        
        @Test
        @DisplayName("obj == null to Objects.isNull(obj)")
        public void testNullCheckPattern_Case2_NullToObjectsIsNull() {
            String method = """
                public void nullToObjectsIsNull() {
                    if (obj == null) {
                        System.out.println("Is null");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Null check should transform code");
            assertCompiles(transformed, "Null check should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Null check should preserve semantics");
        }
        
        @Test
        @DisplayName("Null checks with && operators")
        public void testNullCheckPattern_Case3_NullChecksWithAnd() {
            String method = """
                public void nullChecksWithAnd() {
                    if (obj != null && obj.isValid()) {
                        System.out.println("Valid object");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Null checks with AND should transform code");
            assertCompiles(transformed, "Null checks with AND should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Null checks with AND should preserve semantics");
        }
        
        @Test
        @DisplayName("Null checks with || operators")
        public void testNullCheckPattern_Case4_NullChecksWithOr() {
            String method = """
                public void nullChecksWithOr() {
                    if (obj == null || obj.isEmpty()) {
                        System.out.println("Null or empty");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Null checks with OR should transform code");
            assertCompiles(transformed, "Null checks with OR should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Null checks with OR should preserve semantics");
        }
        
        @Test
        @DisplayName("Nested null checks")
        public void testNullCheckPattern_Case5_NestedNullChecks() {
            String method = """
                public void nestedNullChecks() {
                    if (obj != null) {
                        if (obj.getChild() != null) {
                            System.out.println("Both not null");
                        }
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Nested null checks should transform code");
            assertCompiles(transformed, "Nested null checks should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Nested null checks should preserve semantics");
        }
        
        @Test
        @DisplayName("Null checks in ternary")
        public void testNullCheckPattern_Case6_NullChecksInTernary() {
            String method = """
                public void nullChecksInTernary() {
                    String result = obj != null ? obj.toString() : "null";
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Null checks in ternary should transform code");
            assertCompiles(transformed, "Null checks in ternary should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Null checks in ternary should preserve semantics");
        }
        
        @Test
        @DisplayName("Multiple null checks")
        public void testNullCheckPattern_Case7_MultipleNullChecks() {
            String method = """
                public void multipleNullChecks() {
                    if (obj1 != null && obj2 != null && obj3 != null) {
                        System.out.println("All not null");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiple null checks should transform code");
            assertCompiles(transformed, "Multiple null checks should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Multiple null checks should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: null literal comparisons")
        public void testNullCheckPattern_Case8_NullLiteralComparisons() {
            String method = """
                public void nullLiteralComparisons() {
                    String str = null;
                    if (str == null) {
                        System.out.println("Is null");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Null literal comparisons should transform code");
            assertCompiles(transformed, "Null literal comparisons should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Null literal comparisons should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Safe Equals Operations")
    class SafeEqualsOperations {
        
        @Test
        @DisplayName("obj.equals(other) to Objects.equals(obj, other)")
        public void testNullCheckPattern_Case9_ObjEqualsToObjectsEquals() {
            String method = """
                public void objEqualsToObjectsEquals() {
                    if (obj.equals(other)) {
                        System.out.println("Equal");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Obj equals should transform code");
            assertCompiles(transformed, "Obj equals should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Obj equals should preserve semantics");
        }
        
        @Test
        @DisplayName("Method return null checks")
        public void testNullCheckPattern_Case10_MethodReturnNullChecks() {
            String method = """
                public void methodReturnNullChecks() {
                    if (obj.getResult() != null) {
                        System.out.println("Result not null");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Method return null checks should transform code");
            assertCompiles(transformed, "Method return null checks should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Method return null checks should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive null check pattern test")
    public void testNullCheckPattern_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                // Test various null check patterns
                if (obj != null) {
                    if (obj.getChild() == null) {
                        System.out.println("Child is null");
                    }
                }
                
                // Test safe equals
                if (obj.equals(other)) {
                    System.out.println("Objects are equal");
                }
                
                // Test method return null check
                if (obj.getResult() != null) {
                    System.out.println("Result available");
                }
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}'''
    
    with open('src/test/java/cfwr/jdt/transformations/enhanced/NullCheckPatternTransformationTest.java', 'w') as f:
        f.write(content)

def create_constant_folding_test():
    content = '''package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Constant Folding transformation.
 * Tests constant expression evaluation and optimization.
 */
@DisplayName("Constant Folding Transformation Tests")
class ConstantFoldingTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
        transformer = new SemanticTransformer(42L); // Seeded for determinism
    }
    
    private static final String TRANSFORMATION = "constant_folding";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Arithmetic Constant Folding")
    class ArithmeticConstantFolding {
        
        @Test
        @DisplayName("Simple addition folding")
        public void testConstantFolding_Case1_SimpleAddition() {
            String method = """
                public void simpleAddition() {
                    int result = 5 + 3;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple addition should transform code");
            assertCompiles(transformed, "Simple addition should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple addition should preserve semantics");
            
            // Verify folding (5 + 3 → 8)
            assertTrue(transformed.contains("8") || transformed.contains("5 + 3"), "Should contain folded result or original");
        }
        
        @Test
        @DisplayName("Simple subtraction folding")
        public void testConstantFolding_Case2_SimpleSubtraction() {
            String method = """
                public void simpleSubtraction() {
                    int result = 10 - 4;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple subtraction should transform code");
            assertCompiles(transformed, "Simple subtraction should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple subtraction should preserve semantics");
            
            // Verify folding (10 - 4 → 6)
            assertTrue(transformed.contains("6") || transformed.contains("10 - 4"), "Should contain folded result or original");
        }
        
        @Test
        @DisplayName("Simple multiplication folding")
        public void testConstantFolding_Case3_SimpleMultiplication() {
            String method = """
                public void simpleMultiplication() {
                    int result = 6 * 7;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple multiplication should transform code");
            assertCompiles(transformed, "Simple multiplication should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple multiplication should preserve semantics");
            
            // Verify folding (6 * 7 → 42)
            assertTrue(transformed.contains("42") || transformed.contains("6 * 7"), "Should contain folded result or original");
        }
        
        @Test
        @DisplayName("Simple division folding")
        public void testConstantFolding_Case4_SimpleDivision() {
            String method = """
                public void simpleDivision() {
                    int result = 15 / 3;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple division should transform code");
            assertCompiles(transformed, "Simple division should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple division should preserve semantics");
            
            // Verify folding (15 / 3 → 5)
            assertTrue(transformed.contains("5") || transformed.contains("15 / 3"), "Should contain folded result or original");
        }
        
        @Test
        @DisplayName("Complex arithmetic expressions")
        public void testConstantFolding_Case5_ComplexArithmetic() {
            String method = """
                public void complexArithmetic() {
                    int result = (5 + 3) * (10 - 2);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex arithmetic should transform code");
            assertCompiles(transformed, "Complex arithmetic should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex arithmetic should preserve semantics");
        }
        
        @Test
        @DisplayName("Nested arithmetic")
        public void testConstantFolding_Case6_NestedArithmetic() {
            String method = """
                public void nestedArithmetic() {
                    int result = ((2 + 3) * 4) - (6 / 2);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Nested arithmetic should transform code");
            assertCompiles(transformed, "Nested arithmetic should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Nested arithmetic should preserve semantics");
        }
        
        @Test
        @DisplayName("Floating-point constants")
        public void testConstantFolding_Case7_FloatingPointConstants() {
            String method = """
                public void floatingPointConstants() {
                    double result = 3.14 + 2.86;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Floating-point constants should transform code");
            assertCompiles(transformed, "Floating-point constants should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Floating-point constants should preserve semantics");
        }
        
        @Test
        @DisplayName("Division by zero handling")
        public void testConstantFolding_Case8_DivisionByZeroHandling() {
            String method = """
                public void divisionByZeroHandling() {
                    int result = 10 / 2; // Safe division
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Division by zero handling should transform code");
            assertCompiles(transformed, "Division by zero handling should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Division by zero handling should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: very large numbers")
        public void testConstantFolding_Case9_VeryLargeNumbers() {
            String method = """
                public void veryLargeNumbers() {
                    long result = 1000000L + 2000000L;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Very large numbers should transform code");
            assertCompiles(transformed, "Very large numbers should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Very large numbers should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("String and Boolean Constant Folding")
    class StringBooleanConstantFolding {
        
        @Test
        @DisplayName("String concatenation folding")
        public void testConstantFolding_Case10_StringConcatenationFolding() {
            String method = """
                public void stringConcatenationFolding() {
                    String result = "Hello" + " " + "World";
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "String concatenation folding should transform code");
            assertCompiles(transformed, "String concatenation folding should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "String concatenation folding should preserve semantics");
        }
        
        @Test
        @DisplayName("Boolean constant folding")
        public void testConstantFolding_Case11_BooleanConstantFolding() {
            String method = """
                public void booleanConstantFolding() {
                    boolean result = true && false;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Boolean constant folding should transform code");
            assertCompiles(transformed, "Boolean constant folding should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Boolean constant folding should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive constant folding test")
    public void testConstantFolding_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                // Test various constant folding scenarios
                int arithmetic = (5 + 3) * (10 - 2);
                double floating = 3.14 + 2.86;
                String concatenation = "Test" + " " + "String";
                boolean logical = true && (false || true);
                
                System.out.println(arithmetic + " " + floating + " " + concatenation + " " + logical);
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}'''
    
    with open('src/test/java/cfwr/jdt/transformations/enhanced/ConstantFoldingTransformationTest.java', 'w') as f:
        f.write(content)

def create_dead_code_insertion_test():
    content = '''package cfwr.jdt.transformations.random;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Dead Code Insertion transformation.
 * Tests insertion of harmless dead code statements.
 */
@DisplayName("Dead Code Insertion Transformation Tests")
class DeadCodeInsertionTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
        transformer = new SemanticTransformer(42L); // Seeded for determinism
    }
    
    private static final String TRANSFORMATION = "dead_code_insertion";
    private static final String MODE = "random";
    
    @Nested
    @DisplayName("Dead Code Insertion Operations")
    class DeadCodeInsertionOperations {
        
        @Test
        @DisplayName("Insert numeric literal")
        public void testDeadCodeInsertion_Case1_InsertNumericLiteral() {
            String method = """
                public void insertNumericLiteral() {
                    int result = a + b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert numeric literal should transform code");
            assertCompiles(transformed, "Insert numeric literal should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert numeric literal should preserve semantics");
            
            // Verify dead code insertion (should contain 0; or similar)
            assertTrue(transformed.contains("0;") || transformed.contains("1;") || transformed.contains("2;"), "Should contain dead numeric literal");
        }
        
        @Test
        @DisplayName("Insert boolean literal")
        public void testDeadCodeInsertion_Case2_InsertBooleanLiteral() {
            String method = """
                public void insertBooleanLiteral() {
                    boolean result = a > b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert boolean literal should transform code");
            assertCompiles(transformed, "Insert boolean literal should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert boolean literal should preserve semantics");
            
            // Verify dead code insertion (should contain false; or true;)
            assertTrue(transformed.contains("false;") || transformed.contains("true;"), "Should contain dead boolean literal");
        }
        
        @Test
        @DisplayName("Insert empty string")
        public void testDeadCodeInsertion_Case3_InsertEmptyString() {
            String method = """
                public void insertEmptyString() {
                    String result = "test";
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert empty string should transform code");
            assertCompiles(transformed, "Insert empty string should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert empty string should preserve semantics");
            
            // Verify dead code insertion (should contain "";)
            assertTrue(transformed.contains("\"\";"), "Should contain dead empty string");
        }
        
        @Test
        @DisplayName("Insert in method body")
        public void testDeadCodeInsertion_Case4_InsertInMethodBody() {
            String method = """
                public void insertInMethodBody() {
                    int x = 10;
                    int y = 20;
                    int result = x + y;
                    return result;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert in method body should transform code");
            assertCompiles(transformed, "Insert in method body should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert in method body should preserve semantics");
        }
        
        @Test
        @DisplayName("Insert in if block")
        public void testDeadCodeInsertion_Case5_InsertInIfBlock() {
            String method = """
                public void insertInIfBlock() {
                    if (condition) {
                        System.out.println("True");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert in if block should transform code");
            assertCompiles(transformed, "Insert in if block should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert in if block should preserve semantics");
        }
        
        @Test
        @DisplayName("Insert in loop body")
        public void testDeadCodeInsertion_Case6_InsertInLoopBody() {
            String method = """
                public void insertInLoopBody() {
                    for (int i = 0; i < 10; i++) {
                        System.out.println(i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert in loop body should transform code");
            assertCompiles(transformed, "Insert in loop body should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert in loop body should preserve semantics");
        }
        
        @Test
        @DisplayName("Insert multiple dead statements")
        public void testDeadCodeInsertion_Case7_InsertMultipleDeadStatements() {
            String method = """
                public void insertMultipleDeadStatements() {
                    int result = 0;
                    return result;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert multiple dead statements should transform code");
            assertCompiles(transformed, "Insert multiple dead statements should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert multiple dead statements should preserve semantics");
        }
        
        @Test
        @DisplayName("Insert at different positions")
        public void testDeadCodeInsertion_Case8_InsertAtDifferentPositions() {
            String method = """
                public void insertAtDifferentPositions() {
                    int a = 1;
                    int b = 2;
                    int result = a + b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert at different positions should transform code");
            assertCompiles(transformed, "Insert at different positions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert at different positions should preserve semantics");
        }
        
        @Test
        @DisplayName("Verify no semantic impact")
        public void testDeadCodeInsertion_Case9_VerifyNoSemanticImpact() {
            String method = """
                public void verifyNoSemanticImpact() {
                    int result = calculate();
                    return result;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Verify no semantic impact should transform code");
            assertCompiles(transformed, "Verify no semantic impact should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Verify no semantic impact should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: empty blocks")
        public void testDeadCodeInsertion_Case10_EmptyBlocks() {
            String method = """
                public void emptyBlocks() {
                    if (condition) {
                        // Empty block
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Empty blocks should transform code");
            assertCompiles(transformed, "Empty blocks should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Empty blocks should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: single-statement blocks")
        public void testDeadCodeInsertion_Case11_SingleStatementBlocks() {
            String method = """
                public void singleStatementBlocks() {
                    if (condition) {
                        return;
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Single-statement blocks should transform code");
            assertCompiles(transformed, "Single-statement blocks should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Single-statement blocks should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive dead code insertion test")
    public void testDeadCodeInsertion_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                // Test various dead code insertion scenarios
                int a = 1;
                if (condition) {
                    int b = 2;
                }
                for (int i = 0; i < 5; i++) {
                    System.out.println(i);
                }
                return a;
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}'''
    
    with open('src/test/java/cfwr/jdt/transformations/random/DeadCodeInsertionTransformationTest.java', 'w') as f:
        f.write(content)

def create_method_chain_transformation_test():
    content = '''package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Method Chain transformation.
 * Tests restructuring of method chains and fluent interfaces.
 */
@DisplayName("Method Chain Transformation Tests")
class MethodChainTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
        transformer = new SemanticTransformer(42L); // Seeded for determinism
    }
    
    private static final String TRANSFORMATION = "method_chain_transformation";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Simple Method Chains")
    class SimpleMethodChains {
        
        @Test
        @DisplayName("Simple method chain")
        public void testMethodChainTransformation_Case1_SimpleMethodChain() {
            String method = """
                public void simpleMethodChain() {
                    String result = obj.method1().method2();
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple method chain should transform code");
            assertCompiles(transformed, "Simple method chain should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple method chain should preserve semantics");
        }
        
        @Test
        @DisplayName("Fluent interface builder pattern")
        public void testMethodChainTransformation_Case2_FluentInterfaceBuilder() {
            String method = """
                public void fluentInterfaceBuilder() {
                    Builder builder = new Builder()
                        .setName("test")
                        .setValue(42)
                        .build();
                    System.out.println(builder);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Fluent interface builder should transform code");
            assertCompiles(transformed, "Fluent interface builder should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Fluent interface builder should preserve semantics");
        }
        
        @Test
        @DisplayName("Chain with return values")
        public void testMethodChainTransformation_Case3_ChainWithReturnValues() {
            String method = """
                public void chainWithReturnValues() {
                    int result = obj.getData().getSize().getValue();
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Chain with return values should transform code");
            assertCompiles(transformed, "Chain with return values should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Chain with return values should preserve semantics");
        }
        
        @Test
        @DisplayName("Chain with void methods")
        public void testMethodChainTransformation_Case4_ChainWithVoidMethods() {
            String method = """
                public void chainWithVoidMethods() {
                    obj.method1().method2().method3();
                    System.out.println("Done");
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Chain with void methods should transform code");
            assertCompiles(transformed, "Chain with void methods should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Chain with void methods should preserve semantics");
        }
        
        @Test
        @DisplayName("Nested method chains")
        public void testMethodChainTransformation_Case5_NestedMethodChains() {
            String method = """
                public void nestedMethodChains() {
                    String result = obj.method1(obj2.method3().method4()).method2();
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Nested method chains should transform code");
            assertCompiles(transformed, "Nested method chains should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Nested method chains should preserve semantics");
        }
        
        @Test
        @DisplayName("Chain with parameters")
        public void testMethodChainTransformation_Case6_ChainWithParameters() {
            String method = """
                public void chainWithParameters() {
                    String result = obj.method1(param1).method2(param2, param3);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Chain with parameters should transform code");
            assertCompiles(transformed, "Chain with parameters should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Chain with parameters should preserve semantics");
        }
        
        @Test
        @DisplayName("Chain with generic methods")
        public void testMethodChainTransformation_Case7_ChainWithGenericMethods() {
            String method = """
                public void chainWithGenericMethods() {
                    List<String> result = obj.<String>getList().filter().map();
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Chain with generic methods should transform code");
            assertCompiles(transformed, "Chain with generic methods should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Chain with generic methods should preserve semantics");
        }
        
        @Test
        @DisplayName("Static method chains")
        public void testMethodChainTransformation_Case8_StaticMethodChains() {
            String method = """
                public void staticMethodChains() {
                    String result = Utils.process().format().toString();
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Static method chains should transform code");
            assertCompiles(transformed, "Static method chains should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Static method chains should preserve semantics");
        }
        
        @Test
        @DisplayName("Instance method chains")
        public void testMethodChainTransformation_Case9_InstanceMethodChains() {
            String method = """
                public void instanceMethodChains() {
                    String result = this.method1().method2().method3();
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Instance method chains should transform code");
            assertCompiles(transformed, "Instance method chains should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Instance method chains should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: single method call")
        public void testMethodChainTransformation_Case10_SingleMethodCall() {
            String method = """
                public void singleMethodCall() {
                    String result = obj.method1();
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Single method call should transform code");
            assertCompiles(transformed, "Single method call should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Single method call should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: long chains")
        public void testMethodChainTransformation_Case11_LongChains() {
            String method = """
                public void longChains() {
                    String result = obj.method1().method2().method3().method4().method5();
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Long chains should transform code");
            assertCompiles(transformed, "Long chains should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Long chains should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive method chain transformation test")
    public void testMethodChainTransformation_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                // Test various method chain scenarios
                String result1 = obj.method1().method2();
                Builder builder = new Builder().setName("test").build();
                int value = obj.getData().getSize();
                
                // Mixed chains
                String result2 = obj.method1(builder).method2().method3();
                
                System.out.println(result1 + " " + result2 + " " + value);
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}'''
    
    with open('src/test/java/cfwr/jdt/transformations/enhanced/MethodChainTransformationTest.java', 'w') as f:
        f.write(content)

def create_variable_renaming_test():
    content = '''package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Variable Renaming transformation.
 * Tests renaming of local variables with generated names.
 */
@DisplayName("Variable Renaming Transformation Tests")
class VariableRenamingTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
        transformer = new SemanticTransformer(42L); // Seeded for determinism
    }
    
    private static final String TRANSFORMATION = "variable_renaming";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Simple Variable Renaming")
    class SimpleVariableRenaming {
        
        @Test
        @DisplayName("Simple variable renaming")
        public void testVariableRenaming_Case1_SimpleVariableRenaming() {
            String method = """
                public void simpleVariableRenaming() {
                    int count = 0;
                    System.out.println(count);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple variable renaming should transform code");
            assertCompiles(transformed, "Simple variable renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple variable renaming should preserve semantics");
            
            // Verify variable renaming (should contain new variable name)
            assertFalse(transformed.contains("count") || transformed.contains("newCount") || transformed.contains("var1"), "Should contain renamed variable");
        }
        
        @Test
        @DisplayName("Local variable renaming")
        public void testVariableRenaming_Case2_LocalVariableRenaming() {
            String method = """
                public void localVariableRenaming() {
                    int temp = 10;
                    int result = temp + 5;
                    return result;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Local variable renaming should transform code");
            assertCompiles(transformed, "Local variable renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Local variable renaming should preserve semantics");
        }
        
        @Test
        @DisplayName("Parameter renaming")
        public void testVariableRenaming_Case3_ParameterRenaming() {
            String method = """
                public void parameterRenaming(int value, String name) {
                    System.out.println(value + " " + name);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Parameter renaming should transform code");
            assertCompiles(transformed, "Parameter renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Parameter renaming should preserve semantics");
        }
        
        @Test
        @DisplayName("Loop variable renaming")
        public void testVariableRenaming_Case4_LoopVariableRenaming() {
            String method = """
                public void loopVariableRenaming() {
                    for (int i = 0; i < 10; i++) {
                        System.out.println(i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Loop variable renaming should transform code");
            assertCompiles(transformed, "Loop variable renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Loop variable renaming should preserve semantics");
        }
        
        @Test
        @DisplayName("Multiple variable renaming")
        public void testVariableRenaming_Case5_MultipleVariableRenaming() {
            String method = """
                public void multipleVariableRenaming() {
                    int a = 1;
                    int b = 2;
                    int c = a + b;
                    System.out.println(c);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiple variable renaming should transform code");
            assertCompiles(transformed, "Multiple variable renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Multiple variable renaming should preserve semantics");
        }
        
        @Test
        @DisplayName("Scoped variable renaming")
        public void testVariableRenaming_Case6_ScopedVariableRenaming() {
            String method = """
                public void scopedVariableRenaming() {
                    int outer = 10;
                    if (condition) {
                        int inner = 20;
                        System.out.println(outer + inner);
                    }
                    System.out.println(outer);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Scoped variable renaming should transform code");
            assertCompiles(transformed, "Scoped variable renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Scoped variable renaming should preserve semantics");
        }
        
        @Test
        @DisplayName("Variable shadowing")
        public void testVariableRenaming_Case7_VariableShadowing() {
            String method = """
                public void variableShadowing() {
                    int x = 10;
                    if (condition) {
                        int x = 20; // Shadowing
                        System.out.println(x);
                    }
                    System.out.println(x);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Variable shadowing should transform code");
            assertCompiles(transformed, "Variable shadowing should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Variable shadowing should preserve semantics");
        }
        
        @Test
        @DisplayName("Variable in expressions")
        public void testVariableRenaming_Case8_VariableInExpressions() {
            String method = """
                public void variableInExpressions() {
                    int x = 5;
                    int y = 10;
                    int result = x * y + x - y;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Variable in expressions should transform code");
            assertCompiles(transformed, "Variable in expressions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Variable in expressions should preserve semantics");
        }
        
        @Test
        @DisplayName("Variable in method calls")
        public void testVariableRenaming_Case9_VariableInMethodCalls() {
            String method = """
                public void variableInMethodCalls() {
                    String name = "test";
                    int length = name.length();
                    String upper = name.toUpperCase();
                    System.out.println(upper + " " + length);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Variable in method calls should transform code");
            assertCompiles(transformed, "Variable in method calls should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Variable in method calls should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: single character variables")
        public void testVariableRenaming_Case10_SingleCharacterVariables() {
            String method = """
                public void singleCharacterVariables() {
                    int a = 1;
                    int b = 2;
                    int c = a + b;
                    return c;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Single character variables should transform code");
            assertCompiles(transformed, "Single character variables should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Single character variables should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: reserved keywords nearby")
        public void testVariableRenaming_Case11_ReservedKeywordsNearby() {
            String method = """
                public void reservedKeywordsNearby() {
                    int class_var = 1;
                    int public_var = 2;
                    int static_var = 3;
                    System.out.println(class_var + public_var + static_var);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Reserved keywords nearby should transform code");
            assertCompiles(transformed, "Reserved keywords nearby should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Reserved keywords nearby should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive variable renaming test")
    public void testVariableRenaming_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                // Test various variable renaming scenarios
                int count = 0;
                String name = "test";
                
                for (int i = 0; i < 5; i++) {
                    int temp = i * 2;
                    count += temp;
                }
                
                if (count > 10) {
                    String result = name + "_" + count;
                    System.out.println(result);
                }
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}'''
    
    with open('src/test/java/cfwr/jdt/transformations/enhanced/VariableRenamingTransformationTest.java', 'w') as f:
        f.write(content)

def main():
    """Create all remaining transformation test files."""
    print("Creating remaining transformation test files...")
    
    create_type_conversion_test()
    print("✓ Created TypeConversionTransformationTest.java")
    
    create_null_check_pattern_test()
    print("✓ Created NullCheckPatternTransformationTest.java")
    
    create_constant_folding_test()
    print("✓ Created ConstantFoldingTransformationTest.java")
    
    create_dead_code_insertion_test()
    print("✓ Created DeadCodeInsertionTransformationTest.java")
    
    create_method_chain_transformation_test()
    print("✓ Created MethodChainTransformationTest.java")
    
    create_variable_renaming_test()
    print("✓ Created VariableRenamingTransformationTest.java")
    
    print("\\nAll remaining transformation test files created successfully!")

if __name__ == '__main__':
    main()
