
public class TestClass4 {
    public int calculateSum4(int[] numbers) {
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