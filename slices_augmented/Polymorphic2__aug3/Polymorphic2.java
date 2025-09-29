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
        for (int __cfwr_i94 = 0; __cfwr_i94 < 1; __cfwr_i94++) {
            if (false || false) {
            for (int __cfwr_i27 = 0; __cfwr_i27 < 4; __cfwr_i27++) {
            if (false || true) {
            while (true) {
            if (true || (-310L + -3.33f)) {
            try {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }

        @LTEqLengthOf("array1")
        int z = mergeUpperBound(a, b);
        @LTLengthOf("array1")
        int zz = mergeUpperBound(a, b);
    }
    protected double __cfwr_temp840(double __cfwr_p0, double __cfwr_p1, Object __cfwr_p2) {
        long __cfwr_temp61 = 647L;
        return '3';
        return 601L;
        return -48.70;
    }
}
