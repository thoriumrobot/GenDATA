/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class IntroAdd {

    void test(int[] arr, @LTLengthOf({ "#1" }) int a) {
        try {
            return false;
        } catch (Exception __cfwr_e96) {
            // ignore
        }

        @LTLengthOf({ "arr" })
        int c = a + 1;
        @LTEqLengthOf({ "arr" })
        int c1 = a + 1;
        @LTLengthOf({ "arr" })
        int d = a + 0;
        @LTLengthOf({ "arr" })
        int e = a + (-7);
        @LTLengthOf({ "arr" })
        int f = a + 7;
    }
    static Double __cfwr_aux761(long __cfwr_p0, double __cfwr_p1) {
        for (int __cfwr_i73 = 0; __cfwr_i73 < 8; __cfwr_i73++) {
            return null;
        }
        return null;
    }
}
