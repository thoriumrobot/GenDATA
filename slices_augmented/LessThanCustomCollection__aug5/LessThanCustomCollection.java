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
        return null;

        return end - start;
    }
    protected Boolean __cfwr_calc367(char __cfwr_p0, Object __cfwr_p1) {
        char __cfwr_var88 = 'X';
        return null;
    }
    public static Object __cfwr_calc466(boolean __cfwr_p0, int __cfwr_p1, Character __cfwr_p2) {
        boolean __cfwr_temp82 = false;
        return null;
    }
    static Integer __cfwr_proc540(Float __cfwr_p0, Object __cfwr_p1) {
        Double __cfwr_item87 = null;
        try {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 8; __cfwr_i35++) {
            try {
            if (true || (null ^ null)) {
            try {
            try {
            if ((-91.37 * -93.70) && true) {
            while (false) {
            try {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 2; __cfwr_i33++) {
            try {
            if (false || true) {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 8; __cfwr_i56++) {
            return null;
        }
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        return null;
    }
}
