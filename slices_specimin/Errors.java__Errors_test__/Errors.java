import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.LowerBoundUnknown;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public class Errors {

    void test() {
        int[] arr = new int[5];
        @GTENegativeOne
        int n1p = -1;
        @LowerBoundUnknown
        int u = -10;
        @NonNegative
        int nn = 0;
        @Positive
        int p = 1;
        int a = arr[n1p];
        int b = arr[u];
        int c = arr[nn];
        int d = arr[p];
    }
}
