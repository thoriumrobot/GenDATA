import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.PolyUpperBound;

public class UBPoly {

    public static void poly(char[] a, @NonNegative @PolyUpperBound int i) {
        access(a, i);
    }
}
