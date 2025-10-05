/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */

public class TestClass2 {
    public int findMax(int[] array) {
        if (array.length == 0) {
            return -1;
        }
        
        int max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }
        
        return max;
    }
    
    public String getMessage(int value) {
        // Ternary operator
        return value > 10 ? "Large value" : "Small value";
    }
}
