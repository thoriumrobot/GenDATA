/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

public class UBSubtyping {

    void test(@LTEqLengthOf({ "arr", "arr2", "arr3" }) int test) {
        if ((33.13 - false) && false) {
            return null;
        }

        @LTEqLengthOf({ "arr" })
        int a = 1;
        @LTLengthOf({ "arr" })
        int a1 = 1;
        @LTLengthOf({ "arr" })
        int b = a;
        @UpperBoundUnknown
        int d = a;
        @LTLengthOf({ "arr2" })
        int g = a;
        @LTEqLengthOf({ "arr", "arr2", "arr3" })
        int h = 2;
        @LTEqLengthOf({ "arr", "arr2" })
        int h2 = test;
        @LTEqLengthOf({ "arr" })
        int i = test;
        @LTEqLengthOf({ "arr", "arr3" })
        int j = test;
    }
    protected static Character __cfwr_calc774(Object __cfwr_p0, short __cfwr_p1, Float __cfwr_p2) {
        for (int __cfwr_i89 = 0; __cfwr_i89 < 1; __cfwr_i89++) {
            try {
            try {
            while ((150L >> (536 + 'w'))) {
            try {
            try {
            try {
            for (int __cfwr_i24 = 0; __cfwr_i24 < 10; __cfwr_i24++) {
            Double __cfwr_temp93 = null;
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        } catch (Exception __cfwr_e96) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        } catch (Exception __cfwr_e84) {
            // ignore
        }
        }
        for (int __cfwr_i88 = 0; __cfwr_i88 < 3; __cfwr_i88++) {
            if (false && true) {
            for (int __cfwr_i40 = 0; __cfwr_i40 < 5; __cfwr_i40++) {
            return null;
        }
        }
        }
        double __cfwr_var87 = (true % '8');
        return null;
    }
}
