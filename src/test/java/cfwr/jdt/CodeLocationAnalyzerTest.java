package cfwr.jdt;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

/**
 * Unit tests for CodeLocationAnalyzer
 */
public class CodeLocationAnalyzerTest {
    
    private CodeLocationAnalyzer analyzer;
    
    @BeforeEach
    void setUp() {
        analyzer = new CodeLocationAnalyzer();
    }
    
    @Test
    void testAnalyzeSimpleClass() {
        String javaCode = """
            public class SimpleClass {
                public void method() {
                    int x = 5;
                }
            }
            """;
        
        List<CodeLocation> locations = analyzer.analyzeCode(javaCode);
        
        assertNotNull(locations, "Locations should not be null");
        assertFalse(locations.isEmpty(), "Should find at least one location");
        
        // Check that we find class-level location
        boolean foundClass = locations.stream()
            .anyMatch(loc -> "CLASS_LEVEL".equals(loc.getLocationType()));
        assertTrue(foundClass, "Should find class-level location");
        
        // Check that we find method-level location
        boolean foundMethod = locations.stream()
            .anyMatch(loc -> "METHOD_LEVEL".equals(loc.getLocationType()));
        assertTrue(foundMethod, "Should find method-level location");
    }
    
    @Test
    void testAnalyzeClassWithMultipleMethods() {
        String javaCode = """
            public class MultiMethodClass {
                public void method1() {
                    System.out.println("Method 1");
                }
                
                public void method2(int x) {
                    if (x > 0) {
                        System.out.println("Positive: " + x);
                    }
                }
            }
            """;
        
        List<CodeLocation> locations = analyzer.analyzeCode(javaCode);
        
        assertNotNull(locations, "Locations should not be null");
        assertFalse(locations.isEmpty(), "Should find at least one location");
        
        // Count method-level locations
        long methodCount = locations.stream()
            .filter(loc -> "METHOD_LEVEL".equals(loc.getLocationType()))
            .count();
        assertEquals(2, methodCount, "Should find 2 method-level locations");
        
        // Check that we find statement-level locations
        boolean foundStatement = locations.stream()
            .anyMatch(loc -> "STATEMENT_LEVEL".equals(loc.getLocationType()));
        assertTrue(foundStatement, "Should find statement-level locations");
    }
    
    @Test
    void testAnalyzeClassWithLoops() {
        String javaCode = """
            public class LoopClass {
                public void testLoops() {
                    for (int i = 0; i < 10; i++) {
                        System.out.println(i);
                    }
                    
                    while (true) {
                        break;
                    }
                }
            }
            """;
        
        List<CodeLocation> locations = analyzer.analyzeCode(javaCode);
        
        assertNotNull(locations, "Locations should not be null");
        assertFalse(locations.isEmpty(), "Should find at least one location");
        
        // Check that we find block-level locations (for loops)
        boolean foundBlock = locations.stream()
            .anyMatch(loc -> "BLOCK_LEVEL".equals(loc.getLocationType()));
        assertTrue(foundBlock, "Should find block-level locations");
    }
    
    @Test
    void testAnalyzeClassWithExpressions() {
        String javaCode = """
            public class ExpressionClass {
                public int calculate(int a, int b) {
                    return a + b * 2;
                }
            }
            """;
        
        List<CodeLocation> locations = analyzer.analyzeCode(javaCode);
        
        assertNotNull(locations, "Locations should not be null");
        assertFalse(locations.isEmpty(), "Should find at least one location");
        
        // Check that we find expression-level locations
        boolean foundExpression = locations.stream()
            .anyMatch(loc -> "EXPRESSION_LEVEL".equals(loc.getLocationType()));
        assertTrue(foundExpression, "Should find expression-level locations");
    }
    
    @Test
    void testValidateValidSyntax() {
        String validJavaCode = """
            public class ValidClass {
                public void validMethod() {
                    System.out.println("Valid");
                }
            }
            """;
        
        boolean isValid = analyzer.validateSyntax(validJavaCode);
        assertTrue(isValid, "Valid Java code should pass syntax validation");
    }
    
    @Test
    void testValidateInvalidSyntax() {
        String invalidJavaCode = """
            public class InvalidClass {
                public void invalidMethod() {
                    System.out.println("Invalid"  // Missing closing parenthesis
                }
            }
            """;
        
        boolean isValid = analyzer.validateSyntax(invalidJavaCode);
        assertFalse(isValid, "Invalid Java code should fail syntax validation");
    }
    
    @Test
    void testAnalyzeEmptyCode() {
        String emptyCode = "";
        
        List<CodeLocation> locations = analyzer.analyzeCode(emptyCode);
        
        assertNotNull(locations, "Locations should not be null");
        assertTrue(locations.isEmpty(), "Empty code should return empty locations");
    }
    
    @Test
    void testAnalyzeNullCode() {
        String nullCode = null;
        
        // Should handle null input gracefully without throwing exception
        assertDoesNotThrow(() -> {
            List<CodeLocation> locations = analyzer.analyzeCode(nullCode);
            assertNotNull(locations, "Locations should not be null");
            assertTrue(locations.isEmpty(), "Null code should return empty locations");
        });
    }
    
    @Test
    void testLocationContext() {
        String javaCode = """
            public class ContextClass {
                private String field;
                
                public void method(String param) {
                    int localVar = 42;
                }
            }
            """;
        
        List<CodeLocation> locations = analyzer.analyzeCode(javaCode);
        
        assertNotNull(locations, "Locations should not be null");
        assertFalse(locations.isEmpty(), "Should find at least one location");
        
        // Check that locations have context information
        for (CodeLocation location : locations) {
            assertNotNull(location.getContext(), "Location should have context");
            assertNotNull(location.getLocationType(), "Location should have type");
            assertTrue(location.getLineStart() > 0, "Line start should be positive");
            assertTrue(location.getLineEnd() >= location.getLineStart(), "Line end should be >= line start");
        }
    }
}
