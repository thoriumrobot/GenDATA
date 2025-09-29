/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class LTLDivide {

    void test2(int[] array) {
        Boolean __cfwr_result56 = null;

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
    public static String __cfwr_he
        if (false && true) {
            Character __cfwr_temp77 = null;
        }
lper270(Long __cfwr_p0, byte __cfwr_p1) {
        try {
            if (false && (false * 'c')) {
            return "world94";
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        return "data14";
    }
    protected static int __cfwr_helper17() {
        return null;
        return -907;
    }
}
