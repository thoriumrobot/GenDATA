public class ExampleUsage {

    void obviously_unsafe_loop() {
        int[] arr = new int[5];
        int k;
        for (int i = -1; i < 5; i++) {
            k = arr[i];
        }
    }
}
