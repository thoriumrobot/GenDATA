import org.checkerframework.checker.index.qual.*;

public class IntroAnd {

    void test_ubc_and(@IndexFor("#2") int i, int[] a, @LTLengthOf("#2") int j, int k, @NonNegative int m) {
        int x = a[i & k];
        int x1 = a[k & i];
        int y = a[j & k];
        if (j > -1) {
            int z = a[j & k];
        }
        int w = a[m & k];
        if (m < a.length) {
            int u = a[m & k];
        }
    }
}
