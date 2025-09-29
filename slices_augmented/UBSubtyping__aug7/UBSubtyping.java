/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

public class UBSubtyping {

    void test(@LTEqLengthOf({ "arr", "arr2", "arr3" }) int test) {
        return null;

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
    public static double __cfwr_util98(boolean __cfwr_p0, short __cfwr_p1) {
        for (int __cfwr_i70 = 0; __cfwr_i70 < 8; __cfwr_i70++) {
            while ((187L << -90.35f)) {
            Character __cfwr_result5 = null;
            break; // Prevent infinite loops
        }
        }
        if (false && true) {
            while (false) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return 32.26;
    }
    public static byte __cfwr_handle901(String __cfwr_p0) {
        if (true && false) {
            try {
            while (false) {
            try {
            for (int __cfwr_i83 = 0; __cfwr_i83 < 9; __cfwr_i83++) {
            for (int __cfwr_i46 = 0; __cfwr_i46 < 2; __cfwr_i46++) {
            return null;
        }
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        }
        while (true) {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 7; __cfwr_i17++) {
            Float __cfwr_obj71 = null;
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i26 = 0; __cfwr_i26 < 5; __cfwr_i26++) {
            long __cfwr_node8 = -68L;
        }
        return (-97.85f | 'S');
    }
}
