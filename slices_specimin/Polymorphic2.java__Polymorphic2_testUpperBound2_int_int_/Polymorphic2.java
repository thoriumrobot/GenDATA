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
        @LTEqLengthOf("array1")
        int z = mergeUpperBound(a, b);
        @LTLengthOf("array1")
        int zz = mergeUpperBound(a, b);
    }
}
