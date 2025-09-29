import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.PolyLowerBound;
import org.checkerframework.checker.index.qual.PolySameLen;
import org.checkerframework.checker.index.qual.PolyUpperBound;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.index.qual.SameLen;

public class Polymorphic {

    void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        int[] banana;
        int @SameLen("a2") [] b = samelen_identity(a);
        int @SameLen("banana") [] c = samelen_identity(b);
    }
}
