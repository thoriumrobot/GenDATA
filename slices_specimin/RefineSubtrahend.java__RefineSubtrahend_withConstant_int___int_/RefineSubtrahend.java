import org.checkerframework.checker.index.qual.NonNegative;

public class RefineSubtrahend {

    void withConstant(int[] a, @NonNegative int l) {
        if (a.length - l > 10) {
            int x = a[l + 10];
        }
        if (a.length - 10 > l) {
            int x = a[l + 10];
        }
        if (a.length - l >= 10) {
            int x = a[l + 10];
            int x1 = a[l + 9];
        }
    }
}
