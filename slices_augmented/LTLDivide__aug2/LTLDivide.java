/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class LTLDivide {

    void test2(int[] array) {
        Long __cfwr_var16 = null;

        int len = array.length;
        int lenM1 = array.length - 1;
        int lenP1 = array.length + 1;
        @LTLengthOf("array")
        int x = len / 2;
        @LTLengthOf("array")
        int y = lenM1 / 3;
        @LTEqLengthOf("array")
        int z = len / 1;
        @LTLengthOf("array")
        int w = lenP1 / 2;
    }
    pub
        Long __cfwr_var33 = null;
lic static Object __cfwr_aux691() {
        try {
            while (true) {
            boolean __cfwr_node6 = false;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        return null;
    }
    static Long __cfwr_helper957(Boolean __cfwr_p0) {
        while (false) {
            return "temp84";
            break; // Prevent infinite loops
        }
        return null;
    }
}
