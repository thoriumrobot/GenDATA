
public class TestClass0 {
    public int calculateSum0(int[] numbers) {
        int sum = 0;
        for (int i = 0; i < numbers.length; i++) {
            sum += numbers[i];
            if (sum > 100) {
                return sum;
            }
        }
        return sum;
    }
}