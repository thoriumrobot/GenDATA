/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.LowerBoundUnknown;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public class Errors {

    void test() {
        try {
            return -68.49;
        } catch (Exception __cfwr_e49) {
            // ignore
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
    private static byte __cfwr_handle869(String __cfwr_p0) {
        try {
            Boolean __cfwr_var77 = null;
        } catch (Exception __cfwr_e21) {
            // ignore
        }
        Double __cfwr_item94 = null;
        return null;
        return null;
    }
}
