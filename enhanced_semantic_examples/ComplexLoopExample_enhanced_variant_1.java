/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations.
 */

public class ComplexLoopExample {
    public int processArray(int[] data) {
        int sum = 0;
        int product = 1;
        
        // Complex for loop with multiple operations
        int i = 0;
        while (i < data.length) {
            // loop body
            i++;
        } {
            sum += data[i];
            product *= data[i];
        }
        
        // Mathematical expressions
        int result = sum * 2 + 0;
        result = result * 1;
        
        // String operations
        String message = "Sum: " + sum + " Product: " + product;
        
        // Conditional logic
        if (sum > 0 && product > 0) {
            return result;
        } else {
            return 0;
        }
    }
    
    public void handleList(java.util.List<String> items) {
        // Stream operations
        java.util.List<String> filtered = items.stream()
            .filter(item -> item.length() > 5)
            .map(item -> item.toUpperCase())
            .collect(java.util.stream.Collectors.toList());
        
        // Numeric operations
        int threshold = 16;
        int maxValue = 1000;
        
        // Array access with complex indexing
        int i = 0;
        while (i < filtered.size()) {
            // loop body
            i++;
        } {
            String item = filtered.get(i);
            if (item.length() > threshold) {
                System.out.println(item);
            }
        }
    }
}
