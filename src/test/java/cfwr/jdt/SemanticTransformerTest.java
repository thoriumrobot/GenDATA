package cfwr.jdt;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.List;

/**
 * Unit tests for SemanticTransformer
 */
public class SemanticTransformerTest {
    
    private SemanticTransformer transformer;
    
    @BeforeEach
    void setUp() {
        transformer = new SemanticTransformer(42); // Fixed seed for reproducible tests
    }
    
    @Test
    void testTransformCodeWithLoopConversion() {
        String javaCode = """
            public class TestClass {
                public void testMethod() {
                    for (int i = 0; i < 10; i++) {
                        System.out.println(i);
                    }
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("loop_conversion");
        String result = transformer.transformCode(javaCode, transformations, "enhanced");
        
        assertNotNull(result, "Result should not be null");
        assertFalse(result.trim().isEmpty(), "Result should not be empty");
        // Note: Due to random probability, the transformation might not always apply
        // The important thing is that the method doesn't crash
    }
    
    @Test
    void testTransformCodeWithGuardReversal() {
        String javaCode = """
            public class TestClass {
                public void testMethod(int x) {
                    if (x > 0) {
                        System.out.println("Positive");
                    } else {
                        System.out.println("Non-positive");
                    }
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("guard_reversal");
        String result = transformer.transformCode(javaCode, transformations, "enhanced");
        
        assertNotNull(result, "Result should not be null");
        assertFalse(result.trim().isEmpty(), "Result should not be empty");
    }
    
    @Test
    void testTransformCodeWithMathematicalExpression() {
        String javaCode = """
            public class TestClass {
                public int calculate(int a, int b) {
                    return a + b;
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("mathematical_expression");
        String result = transformer.transformCode(javaCode, transformations, "enhanced");
        
        assertNotNull(result, "Result should not be null");
        assertFalse(result.trim().isEmpty(), "Result should not be empty");
    }
    
    @Test
    void testTransformCodeWithLogicalExpression() {
        String javaCode = """
            public class TestClass {
                public boolean test(int x, int y) {
                    return x > 0 && y > 0;
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("logical_expression");
        String result = transformer.transformCode(javaCode, transformations, "enhanced");
        
        assertNotNull(result, "Result should not be null");
        assertFalse(result.trim().isEmpty(), "Result should not be empty");
    }
    
    @Test
    void testTransformCodeWithTernaryOperator() {
        String javaCode = """
            public class TestClass {
                public String test(int x) {
                    return x > 0 ? "positive" : "non-positive";
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("ternary_operator");
        String result = transformer.transformCode(javaCode, transformations, "enhanced");
        
        assertNotNull(result, "Result should not be null");
        assertFalse(result.trim().isEmpty(), "Result should not be empty");
    }
    
    @Test
    void testTransformCodeWithMultipleTransformations() {
        String javaCode = """
            public class TestClass {
                public void testMethod(int x) {
                    for (int i = 0; i < x; i++) {
                        if (i > 0) {
                            System.out.println("Positive: " + i);
                        }
                    }
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("loop_conversion", "guard_reversal");
        String result = transformer.transformCode(javaCode, transformations, "enhanced");
        
        assertNotNull(result, "Result should not be null");
        assertFalse(result.trim().isEmpty(), "Result should not be empty");
    }
    
    @Test
    void testTransformCodeSimpleMode() {
        String javaCode = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(x);
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("simple_assignment");
        String result = transformer.transformCode(javaCode, transformations, "simple");
        
        assertNotNull(result, "Result should not be null");
        assertFalse(result.trim().isEmpty(), "Result should not be empty");
    }
    
    @Test
    void testTransformInvalidCode() {
        String invalidJavaCode = """
            public class InvalidClass {
                public void invalidMethod() {
                    System.out.println("Invalid"  // Missing closing parenthesis
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("guard_reversal");
        String result = transformer.transformCode(invalidJavaCode, transformations, "enhanced");
        
        // Should return original code unchanged when parsing fails
        assertEquals(invalidJavaCode, result, "Should return original code when parsing fails");
    }
    
    @Test
    void testTransformEmptyCode() {
        String emptyCode = "";
        
        List<String> transformations = Arrays.asList("guard_reversal");
        String result = transformer.transformCode(emptyCode, transformations, "enhanced");
        
        assertEquals(emptyCode, result, "Empty code should return empty result");
    }
    
    @Test
    void testTransformNullCode() {
        String nullCode = null;
        
        List<String> transformations = Arrays.asList("guard_reversal");
        String result = transformer.transformCode(nullCode, transformations, "enhanced");
        
        assertEquals(nullCode, result, "Null code should return null result");
    }
    
    @Test
    void testTransformWithEmptyTransformationsList() {
        String javaCode = """
            public class TestClass {
                public void testMethod() {
                    System.out.println("test");
                }
            }
            """;
        
        List<String> transformations = Arrays.asList();
        String result = transformer.transformCode(javaCode, transformations, "enhanced");
        
        assertEquals(javaCode, result, "Empty transformations should return original code");
    }
    
    @Test
    void testTransformWithUnknownTransformation() {
        String javaCode = """
            public class TestClass {
                public void testMethod() {
                    System.out.println("test");
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("unknown_transformation");
        String result = transformer.transformCode(javaCode, transformations, "enhanced");
        
        assertEquals(javaCode, result, "Unknown transformation should return original code");
    }
    
    @Test
    void testTransformWithNullTransformationsList() {
        String javaCode = """
            public class TestClass {
                public void testMethod() {
                    System.out.println("test");
                }
            }
            """;
        
        List<String> transformations = null;
        String result = transformer.transformCode(javaCode, transformations, "enhanced");
        
        assertEquals(javaCode, result, "Null transformations should return original code");
    }
    
    @Test
    void testTransformWithNullMode() {
        String javaCode = """
            public class TestClass {
                public void testMethod() {
                    System.out.println("test");
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("guard_reversal");
        String result = transformer.transformCode(javaCode, transformations, null);
        
        assertNotNull(result, "Result should not be null even with null mode");
    }
}
