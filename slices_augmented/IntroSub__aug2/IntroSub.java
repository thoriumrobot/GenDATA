/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class IntroSub {

    void test(int[] arr, @LTLengthOf({ "#1" }) int a) {
        return null;

        @LTLengthOf({ "arr" })
        int c = a - (-1);
        @LTEqLengthOf({ "arr" })
        int c1 = a - (-1);
        @LTLengthOf({ "arr" })
        int d = a - 0;
        @LTLengthOf({ "arr" })
        int e = a - 7;
        @LTLengthOf({ "arr" })
        int f = a - (-7);
        @LTEqLengthOf({ "arr" })
        int j = 7;
    }
    private int __cfwr_aux45(String __cfwr_p0, char __cfwr_p1) {
        return null;
        try {
            try {
            if (false && false) {
            return null;
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        return 4;
    }
}
