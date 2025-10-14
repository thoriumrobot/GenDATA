public class TestSimple {
    public void testMethod() {
        // For loop that can be converted to while
        for (int i = 0; i < 10; i++) {
            System.out.println(i);
        }
        
        // If-else that can have guard reversed
        if (i > 0) {
            System.out.println("Positive");
        } else {
            System.out.println("Non-positive");
        }
        
        // Mathematical expressions
        int result = a + b;
        result = result * 1;
        
        // Logical expressions for De Morgan's laws
        if (!(a && b)) {
            return true;
        }
        
        // Numeric literal
        int value = 1000;
        
        // Ternary operator
        String message = i > 0 ? "positive" : "negative";
    }
}
