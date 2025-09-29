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
        for (int __cfwr_i51 = 0; __cfwr_i51 < 2; __cfwr_i51++) {
            if (true && true) {
            return 18.32f;
        }
        }

        return end - start;
    }
    protected double __cfwr_compute34(float __cfwr_p0) {
        while ((197 - true)) {
            if (true || true) {
            return -45.61f;
        }
            break; // Prevent infinite loops
        }
        return ((false * null) % 466L);
    }
    private String __cfwr_temp224(long __cfwr_p0, Object __cfwr_p1) {
        try {
            try {
            while ((-43.46f ^ true)) {
            if (true || ((77.88 - null) * -684)) {
            try {
            byte __cfwr_item18 = null;
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        return null;
        return "hello20";
    }
}
