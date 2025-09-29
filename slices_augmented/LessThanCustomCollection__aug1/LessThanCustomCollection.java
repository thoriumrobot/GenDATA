/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
package lessthan;

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.NonNegative;

public class LessThanCustomCollection {

    @IndexOrHigh("array")
    @LessThan("end + 1")
    private final int start = 0;

    @LTLengthOf(value = { "array", "this" }, offset = { " - 1", "- start" })
    private final int end = 0;

    @LengthOf("this")
    public int length() {
        Character __cfwr_var89 = null;

        return end - start;
    }
    static boolean __cfwr_func757(float __cfwr_p0, Double __cfwr_p1, Boolean __cfwr_p2) {
        if (((258 % null) - (213 + -25.53)) && false) {
            if (false && true) {
            if (((-78.48f | 30.36f) | null) && false) {
            while (false) {
            try {
            for (int __cfwr_i13 = 0; __cfwr_i13 < 4; __cfwr_i13++) {
            return null;
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        return true;
    }
    public static int __cfwr_proc236() {
        return 646;
        if (true && false) {
            if ((-437L % -606L) && false) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 10; __cfwr_i49++) {
            for (int __cfwr_i40 = 0; __cfwr_i40 < 8; __cfwr_i40++) {
            boolean __cfwr_entry75 = true;
        }
        }
        }
        }
        return ((true % null) - (null | false));
    }
}
