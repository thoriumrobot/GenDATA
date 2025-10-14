public class TestFile {
    private int[] array = new int[10];
    
    public int getValue(int index) {
        return array[index];  // Potential out-of-bounds access
    }
    
    public void setValue(int index, int value) {
        array[index] = value;
    }
}