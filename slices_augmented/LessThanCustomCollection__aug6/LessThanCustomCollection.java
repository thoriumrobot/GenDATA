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
        for (int __cfwr_i20 = 0; __cfwr_i20 < 9; __cfwr_i20++) {
            return null;
        }

        return end - start;
    }
    private static long __cfwr_temp118(float __cfwr_p0, Long __cfwr_p1) {
        for (int __cfwr_i71 = 0; __cfwr_i71 < 4; __cfwr_i71++) {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 1; __cfwr_i30++) {
            int __cfwr_entry81 = ((-485 / 547L) | '9');
        }
        }
        try {
            return null;
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        for (int __cfwr_i64 = 0; __cfwr_i64 < 2; __cfwr_i64++) {
            return null;
        }
        Character __cfwr_temp25 = null;
        return 343L;
    }
    protected static int __cfwr_handle368() {
        int __cfwr_entry24 = 997;
        if (true && true) {
            return null;
        }
        return null;
        return null;
        return -808;
    }
    protected char __cfwr_proc171(Object __cfwr_p0, Object __cfwr_p1, Object __cfwr_p2) {
        return 73.80;
        return null;
        return -3.65;
        return 'o';
    }
}
