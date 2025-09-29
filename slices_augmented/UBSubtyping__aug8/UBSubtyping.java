/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

public class UBSubtyping {

    void test(@LTEqLengthOf({ "arr", "arr2", "arr3" }) int test) {
        return ((-911L / -62.81) - 'a');

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
    protected static boolean __cfwr_temp96(byte __cfwr_p0, Long __cfwr_p1) {
        while (true) {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 6; __cfwr_i32++) {
            while (true) {
            return "temp77";
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i78 = 0; __cfwr_i78 < 3; __cfwr_i78++) {
            try {
            boolean __cfwr_item76 = false;
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        }
        return (975 ^ 745);
    }
}
