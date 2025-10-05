/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */

public class TestClass1 {
    public int calculateSum(int[] array) {
        int sum = 0;
        
        // For loop that can be converted to while
        int i = 0;
        while (i < array.length) {
            sum += array[i];
            i++;
        }
        
        // If-else that can have guard reversed
        if (sum <= 0) {
            System.out.println("Non-positive sum");
        } else {
            System.out.println("Positive sum");
        }
        
        // Mathematical expressions
        int result = sum << 1;
        result = result;
        
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
