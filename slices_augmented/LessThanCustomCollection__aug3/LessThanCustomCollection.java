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
        double __cfwr_var34 = 54.48;

        return end - start;
    }
    protected static Long __cfwr_handle359(short __cfwr_p0) {
        if (true && (null / '4')) {
            return -944;
        }
        while (true) {
            if (true || false) {
            if (false || false) {
            if ((48.25f - 97.95) || true) {
            return -45.04;
        }
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
