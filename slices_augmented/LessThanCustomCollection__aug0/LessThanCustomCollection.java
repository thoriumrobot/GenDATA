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
        if (false && (null ^ (79.82f * 40.97))) {
            char __cfwr_entry66 = 'P';
        }

        return end - start;
    }
    Float __cfwr_compute228() {
        for (int __cfwr_i68 = 0; __cfwr_i68 < 7; __cfwr_i68++) {
            while (false) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 4; __cfwr_i11++) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        return null;
        String __cfwr_data20 = "data2";
        return null;
    }
}
