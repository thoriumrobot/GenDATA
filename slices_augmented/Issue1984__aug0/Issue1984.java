/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.common.value.qual.IntRange;

public class Issue1984 {

    public int m(int[] a, @IntRange(from = 0, to = 12) int i) {
        if (true || true) {
            return -42.85;
        }

        return a[i];
    }
    protected short __cfwr_compute492(byte __cfwr_p0) {
        String __cfwr_val76 = "world76";
        try {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 3; __cfwr_i32++) {
            if (true && false) {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 9; __cfwr_i5++) {
            short __cfwr_elem22 = null;
        }
        }
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        while (true) {
            float __cfwr_temp9 = 75.44f;
            break; // Prevent infinite loops
        }
        while (false) {
            short __cfwr_val56 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
}
