package cfwr.jdt.transformations.enhanced;

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
}