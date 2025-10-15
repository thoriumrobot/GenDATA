package cfwr.jdt.transformations.enhanced;

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
}