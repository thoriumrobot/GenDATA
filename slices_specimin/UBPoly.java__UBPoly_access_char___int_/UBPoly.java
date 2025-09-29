import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.PolyUpperBound;

public class UBPoly {

    public static void access(char[] a, @NonNegative @LTLengthOf("#1") int j) {
        char c = a[j];
    }
}
