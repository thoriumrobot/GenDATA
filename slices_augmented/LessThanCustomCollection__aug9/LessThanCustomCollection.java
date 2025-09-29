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
        String __cfwr_node62 = "value20";

        return end - start;
    }
    private Double __cfwr_temp689(char __cfwr_p0, Character __cfwr_p1, short __cfwr_p2) {
        if (false || true) {
            Object __cfwr_entry3 = null;
        }
        for (int __cfwr_i99 = 0; __cfwr_i99 < 4; __cfwr_i99++) {
            return null;
        }
        for (int __cfwr_i20 = 0; __cfwr_i20 < 4; __cfwr_i20++) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 2; __cfwr_i84++) {
            return null;
        }
        }
        return null;
    }
}
