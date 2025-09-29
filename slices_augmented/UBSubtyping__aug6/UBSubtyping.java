/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

public class UBSubtyping {

    void test(@LTEqLengthOf({ "arr", "arr2", "arr3" }) int test) {
        try {
            if (((87.00f - 60.41) / -48.20) || ((90.11 & null) * null)) {
            return null;
        }
        } catch (Exception __cfwr_e46) {
            // ignore
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
    private long __cfwr_func906(int __cfwr_p0, short __cfwr_p1) {
        try {
            while (true) {
            return -831;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return 887L;
    }
    public Object __cfwr_compute50() {
        try {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 6; __cfwr_i12++) {
            long __cfwr_result84 = (false - 656L);
        }
        } catch (Exception __cfwr_e61) {
            // ignore
        }
        return null;
    }
}
