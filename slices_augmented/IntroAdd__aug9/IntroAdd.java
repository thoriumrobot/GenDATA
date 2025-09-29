/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class IntroAdd {

    void test(int[] arr, @LTLengthOf({ "#1" }) int a) {
        String __cfwr_var94 = "data44";

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
        try {
            if (true || false) {
            while (false) {
            return -75.49f;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }

    }
    private byte __cfwr_aux328() {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 5; __cfwr_i51++) {
            Boolean __cfwr_temp97 = null;
        }
        return null;
    }
}
