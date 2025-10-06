/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations.
 */

public class MathematicalOperations {
    public int calculate(int a, int b, int c) {
        // Multiple mathematical operations
        int sum = a + b + c;
        int product = a * b * c;
        int difference = sum - product;
        
        // Complex expressions
        int result = (sum * 2) + (product >> 1);
        result = result * 1 + 0;
        
        // Conditional mathematical operations
        if (a <= b) {
            result = result >> 1;
        } else {
            result = result * 2;
        }
        
        // String concatenation with numbers
        String output = "Result: " + result + " from " + a + " + " + b + " + " + c;
        
        return result;
    }
    
    public boolean compareValues(int x, int y, int z) {
        // Complex boolean logic
        boolean condition1 = (x > y) && (y > z);
        boolean condition2 = (x + y) > (z * 2);
        boolean condition3 = (x * y) < (z + 10);
        
        return condition1 || condition2 || condition3;
    }
}
