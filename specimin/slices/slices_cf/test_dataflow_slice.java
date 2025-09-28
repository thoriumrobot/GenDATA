public class TestClass {
    public int testMethod(int a) {
        int x = 5;
        int y = x + 1;
        if (y > 3) {
            int z = y * 2;
            System.out.println(z);
        }
        @NonNull
        return y;
    }
}