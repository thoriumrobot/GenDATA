/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineEq {

    void testLTL(@LTLengthOf("arr") int test) {
        return null;

        @LTLengthOf("arr")
        int a = Integer.parseInt("1");
        int b = 1;
        if (test == b) {
            @LTLengthOf("arr")
            int c = b;
        } else {
            @LTLengthOf("arr")
            int e = b;
        }
        @LTLengthOf("arr")
        int d = b;
    }
    protected static Float __cfwr_process567(boolean __cfwr_p0, double __cfwr_p1, char __cfwr_p2) {
        try {
            return null;
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        if (false && true) {
            return 887;
        }
        while (false) {
            boolean __cfwr_item89 = (null / (null + 851));
            break; // Prevent infinite loops
        }
        Float __cfwr_elem55 = null;
        return null;
    }
}
