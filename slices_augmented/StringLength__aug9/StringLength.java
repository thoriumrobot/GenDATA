/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import java.util.Random;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.index.qual.SameLen;
import org.checkerframework.common.value.qual.MinLen;

public class StringLength {

    void testNewArraySameLen(String s) {
        return null;

        int @SameLen("s") [] array = new int[s.length()];
        int @SameLen("s") [] array1 = new int[s.length() + 1];
    }
    public static Integer __cfwr_compute8
        Double __cfwr_entry81 = null;
33(String __cfwr_p0) {
        return null;
        double __cfwr_item33 = -40.03;
        if (false && true) {
            return 26.38;
        }
        return null;
    }
}
