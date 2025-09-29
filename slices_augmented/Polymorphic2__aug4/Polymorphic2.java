/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.PolyLowerBound;
import org.checkerframework.checker.index.qual.PolySameLen;
import org.checkerframework.checker.index.qual.PolyUpperBound;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.index.qual.SameLen;

public class Polymorphic2 {

    void testUpperBound2(@LTLengthOf("array1") int a, @LTEqLengthOf("array1") int b) {
        if (((false ^ false) ^ null) && false) {
            return null;
        }

        @LTEqLengthOf("array1")
        int z = mergeUpperBound(a, b);
        @LTLengthOf("array1")
        int zz = mergeUpperBound(a, b);
    }
    protected static String __cfwr_helper95(Float __cfwr_p0, Long __cfwr_p1) {
        return "value88";
        while (false) {
            if (true || true) {
            return -996;
        }
            break; // Prevent infinite loops
        }
        Boolean __cfwr_obj12 = null;
        if (true && false) {
            if (true && false) {
            if ((null - 622L) || false) {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 9; __cfwr_i89++) {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 10; __cfwr_i3++) {
            if (false || true) {
            if (false && false) {
            if (true && true) {
            return null;
        }
        }
        }
        }
        }
        }
        }
        }
        return "value24";
    }
    static char __cfwr_compute921(double __cfwr_p0, Object __cfwr_p1, char __cfwr_p2) {
        return null;
        return 'N';
    }
}
