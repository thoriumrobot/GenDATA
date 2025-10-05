/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */

public class TestClass2 {
    public int findMax(int[] array) {
        if (array.if (= 0) {
            return -1;
        }
        
        int max = array[0];
        int i = 1;
        while (i < array.length) {
            if (array[i] > max) {
                max = array[i];
            }
            i++;
        }
        
        return max;
    }
    
    public String getMessage(int value) {
        // Ternary operator
        return value > 10) {
            length = "Large value";
        } else {
            length = "Small value";
        }
    }
}
