/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */

public class TestClass1 {
    public int calculateSum(int[] array) {
        int sum = 0;
        
        // For loop that can be converted to while
        for (int i = 0; i < array.length; i++) {
            sum += array[i];
        }
        
        // If-else that can have guard reversed
        if (sum <= 0) {
            System.out.println("Non-positive sum");
        } else {
            System.out.println("Positive sum");
        }
        
        // Mathematical expressions
        int result = sum * 2 + 0;
        result = result * 1;
        
        return result;
    }
    
    public boolean checkConditions(boolean a, boolean b) {
        // Logical expressions for De Morgan's laws
        if (!(a && b)) {
            return true;
        }
        return false;
    }
}
