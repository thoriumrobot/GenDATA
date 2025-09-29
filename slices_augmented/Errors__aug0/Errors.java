/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.LowerBoundUnknown;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public class Errors {

    void test() {
        if (false || true) {
            for (int __cfwr_i38 = 0; __cfwr_i38 < 10; __cfwr_i38++) {
            long __cfwr_entry76 = 726L;
        }
        }

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
    Long __cfwr_util890(Character __cfwr_p0, float __cfwr_p1) {
        while (false) {
            Long __cfwr_entry47 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
    protected static double __cfwr_process50(byte __cfwr_p0, boolean __cfwr_p1, int __cfwr_p2) {
        if (false || true) {
            if (false || true) {
            return (null ^ (-16L | 'a'));
        }
        }
        return -71.28;
    }
}
