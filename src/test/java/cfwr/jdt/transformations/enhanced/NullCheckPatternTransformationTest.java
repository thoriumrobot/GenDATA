package cfwr.jdt.transformations.enhanced;

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
}