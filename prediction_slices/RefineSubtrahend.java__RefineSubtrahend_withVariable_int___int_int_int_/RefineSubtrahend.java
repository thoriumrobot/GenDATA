import org.checkerframework.checker.index.qual.NonNegative;

public class RefineSubtrahend {

    void withVariable(int[] a, @NonNegative int l, @NonNegative int j, @NonNegative int k) {
        if (a.length - l > j) {
            if (k <= j) {
                int x = a[l + k];
            }
        }
        if (a.length - j > l) {
            if (k <= j) {
                int x = a[l + k];
            }
        }
        if (a.length - j >= l) {
            if (k <= j) {
                int x = a[l + k];
                int x1 = a[l + k - 1];
            }
        }
    }
}
