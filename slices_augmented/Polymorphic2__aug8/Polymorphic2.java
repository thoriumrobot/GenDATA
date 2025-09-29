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
        Float __cfwr_var81 = null;

        @LTEqLengthOf("array1")
        int z = mergeUpperBound(a, b);
        @LTLengthOf("array1")
        int zz = mergeUpperBound(a, b);
    }
    public int __cfwr_temp217(int __cfwr_p0) {
        int __cfwr_obj62 = 448;
        while (false) {
            try {
            try {
            while (false) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 10; __cfwr_i74++) {
            try {
            return null;
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        if (false && (61.30f + -893L)) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        return "result82";
        return 164;
    }
    static long __cfwr_handle732() {
        return null;
        if (false && true) {
            return "world56";
        }
        return 456L;
    }
}
