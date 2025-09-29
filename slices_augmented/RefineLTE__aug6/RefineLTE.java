/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class RefineLTE {

    void testLTL(@LTLengthOf("arr") int test) {
        return null;

        @LTLengthOf("arr")
        int a = Integer.parseInt("1");
        @LTLengthOf("arr")
        int a3 = Integer.parseInt("3");
        int b = 2;
        if (b <= test) {
            @LTLengthOf("arr")
            int c = b;
        }
        @LTLengthOf("arr")
        int c1 = b;
        if (b <= a) {
            int potato = 7;
        } else {
            @LTLengthOf("arr")
            int d = b;
        }
    }
    protected Character __cfwr_process420(Object __cfwr_p0, boolean __cfwr_p1, long __cfwr_p2) {
        short __cfwr_val50 = (null & (687L << 45.08));
        return null;
        return null;
    }
}
