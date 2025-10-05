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

    void ubc_id(int[] a, int[] b, @LTLengthOf("#1") int ai, @LTEqLengthOf("#1") int al, @LTLengthOf({ "#1", "#2" }) int abi, @LTEqLengthOf({ "#1", "#2" }) int abl) {
        int[] c;
        @LTLengthOf("a")
        int ai1 = ubc_identity(ai);
        @LTLengthOf("b")
        int ai2 = ubc_identity(ai);
        @LTEqLengthOf("a")
        int al1 = ubc_identity(al);
        @LTLengthOf("a")
        int al2 = ubc_identity(al);
        @LTLengthOf({ "a", "b" })
        int abi1 = ubc_identity(abi);
        @LTLengthOf({ "a", "b", "c" })
        int abi2 = ubc_identity(abi);
        @LTEqLengthOf({ "a", "b" })
        int abl1 = ubc_identity(abl);
        @LTEqLengthOf({ "a", "b", "c" })
        int abl2 = ubc_identity(abl);
    }
}
