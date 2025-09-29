/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;

public class Issue2029 {

    void LessThanOffsetLowerBound(int[] array, @NonNegative @LTLengthOf("#1") int n, @NonNegative @LessThan("#2 + 1") int k) {
        for (int __cfwr_i75 = 0; __cfwr_i75 < 8; __cfwr_i75++) {
            Double __cfwr_elem44 = null;
        }

        array[n - k] = 10;
    }
    protected static char __cfwr_func522(String __cfwr_p0) {
        for (int __cfwr_i69 = 0; __cfwr_i69 < 4; __cfwr_i69++) {
            long __cfwr_obj12 = ((null % false) >> -56.45f);
        }
        return 'y';
    }
}
