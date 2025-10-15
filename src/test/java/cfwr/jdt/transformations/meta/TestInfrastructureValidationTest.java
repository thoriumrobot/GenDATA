package cfwr.jdt.transformations.meta;

import cfwr.jdt.transformations.utils.CompilationValidator;
import cfwr.jdt.transformations.utils.SemanticEquivalenceChecker;
import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Meta-tests to validate that the test infrastructure itself works correctly.
 * These tests ensure that our testing utilities are functioning properly.
 */
public class TestInfrastructureValidationTest extends TransformationTestBase {
    
    private CompilationValidator compilationValidator;
    private SemanticEquivalenceChecker equivalenceChecker;
    
    @BeforeEach
    public void setUp() {
        super.setUp();
        compilationValidator = new CompilationValidator();
        equivalenceChecker = new SemanticEquivalenceChecker();
    }
    
    @Test
    public void testCompilationValidator_ValidCode() {
        // Test that valid Java code is recognized as compilable
        String validCode = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(x);
                }
            }
            """;
        
        CompilationValidator.CompilationResult result = compilationValidator.compile(validCode);
        assertTrue(result.isSuccess(), "Valid Java code should compile successfully");
        assertTrue(result.getErrors().isEmpty(), "Valid code should have no compilation errors");
    }
    
    @Test
    public void testCompilationValidator_InvalidCode() {
        // Test that invalid Java code is recognized as non-compilable
        String invalidCode = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(y); // y is undefined
                }
            }
            """;
        
        CompilationValidator.CompilationResult result = compilationValidator.compile(invalidCode);
        assertFalse(result.isSuccess(), "Invalid Java code should not compile");
        assertFalse(result.getErrors().isEmpty(), "Invalid code should have compilation errors");
        assertTrue(result.getErrorOutput().contains("cannot find symbol"), 
                  "Error message should indicate undefined variable");
    }
    
    @Test
    public void testCompilationValidator_ErrorMessages() {
        // Test that error messages are accurate and informative
        String invalidCode = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(y);
                }
            }
            """;
        
        CompilationValidator.CompilationResult result = compilationValidator.compile(invalidCode);
        String errorOutput = result.getErrorOutput();
        
        assertTrue(errorOutput.contains("cannot find symbol"), 
                  "Error should mention 'cannot find symbol'");
        assertTrue(errorOutput.contains("y"), 
                  "Error should mention the undefined variable 'y'");
        assertTrue(errorOutput.contains("symbol:   variable y"), 
                  "Error should specify that 'y' is a variable");
    }
    
    @Test
    public void testSemanticEquivalenceChecker_IdenticalCode() {
        // Test that identical code is recognized as equivalent
        String code1 = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(x);
                }
            }
            """;
        
        String code2 = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(x);
                }
            }
            """;
        
        boolean equivalent = equivalenceChecker.areEquivalent(code1, code2);
        assertTrue(equivalent, "Identical code should be recognized as semantically equivalent");
    }
    
    @Test
    public void testSemanticEquivalenceChecker_DifferentCode() {
        // Test that semantically different code is recognized as non-equivalent
        String code1 = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(x);
                }
            }
            """;
        
        String code2 = """
            public class TestClass {
                public void testMethod() {
                    int x = 10; // Different value
                    System.out.println(x);
                }
            }
            """;
        
        boolean equivalent = equivalenceChecker.areEquivalent(code1, code2);
        assertFalse(equivalent, "Semantically different code should not be recognized as equivalent");
    }
    
    @Test
    public void testSemanticEquivalenceChecker_WhitespaceDifferences() {
        // Test that whitespace differences don't affect semantic equivalence
        String code1 = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(x);
                }
            }
            """;
        
        String code2 = """
            public class TestClass{
                public void testMethod(){
                    int x=5;
                    System.out.println(x);
                }
            }
            """;
        
        boolean equivalent = equivalenceChecker.areEquivalent(code1, code2);
        assertTrue(equivalent, "Code with different whitespace should be semantically equivalent");
    }
    
    @Test
    public void testSemanticEquivalenceChecker_ParenthesesDifferences() {
        // Test that parentheses differences don't affect semantic equivalence
        String code1 = """
            public class TestClass {
                public void testMethod() {
                    int x = (5 + 3) * 2;
                    System.out.println(x);
                }
            }
            """;
        
        String code2 = """
            public class TestClass {
                public void testMethod() {
                    int x = 5 + 3 * 2; // Different parentheses, same semantics
                    System.out.println(x);
                }
            }
            """;
        
        boolean equivalent = equivalenceChecker.areEquivalent(code1, code2);
        assertTrue(equivalent, "Code with different parentheses but same semantics should be equivalent");
    }
    
    @Test
    public void testTransformationTestBase_AssertCompiles() {
        // Test that assertCompiles works correctly
        String validCode = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(x);
                }
            }
            """;
        
        // This should not throw an exception
        assertDoesNotThrow(() -> assertCompiles(validCode, "Valid code test"));
    }
    
    @Test
    public void testTransformationTestBase_AssertCompilesFailure() {
        // Test that assertCompiles throws exception for invalid code
        String invalidCode = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(y); // y is undefined
                }
            }
            """;
        
        // This should throw an AssertionError
        assertThrows(AssertionError.class, () -> 
            assertCompiles(invalidCode, "Invalid code test"));
    }
    
    @Test
    public void testTransformationTestBase_AssertSemanticallyEquivalent() {
        // Test that assertSemanticallyEquivalent works correctly
        String code1 = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(x);
                }
            }
            """;
        
        String code2 = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(x);
                }
            }
            """;
        
        // This should not throw an exception
        assertDoesNotThrow(() -> 
            assertSemanticallyEquivalent(code1, code2, "Equivalent code test"));
    }
    
    @Test
    public void testTransformationTestBase_AssertSemanticallyEquivalentFailure() {
        // Test that assertSemanticallyEquivalent throws exception for different code
        String code1 = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    System.out.println(x);
                }
            }
            """;
        
        String code2 = """
            public class TestClass {
                public void testMethod() {
                    int x = 10; // Different value
                    System.out.println(x);
                }
            }
            """;
        
        // This should throw an AssertionError
        assertThrows(AssertionError.class, () -> 
            assertSemanticallyEquivalent(code1, code2, "Different code test"));
    }
    
    @Test
    public void testTransformationTestBase_AssertTransformationApplied() {
        // Test that assertTransformationApplied works correctly
        String original = "int x = 5;";
        String transformed = "int x = (5);";
        
        // This should not throw an exception
        assertDoesNotThrow(() -> 
            assertTransformationApplied(original, transformed, "Transformation applied test"));
    }
    
    @Test
    public void testTransformationTestBase_AssertTransformationAppliedFailure() {
        // Test that assertTransformationApplied throws exception when no transformation
        String original = "int x = 5;";
        String transformed = "int x = 5;"; // Same code
        
        // This should throw an AssertionError
        assertThrows(AssertionError.class, () -> 
            assertTransformationApplied(original, transformed, "No transformation test"));
    }
}
