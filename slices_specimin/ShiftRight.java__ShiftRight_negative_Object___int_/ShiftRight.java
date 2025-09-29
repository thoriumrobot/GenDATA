import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class ShiftRight {

    void negative(Object[] a, @LTLengthOf(value = "#1", offset = "100") int i) {
        @LTLengthOf(value = "#1", offset = "100")
        int q = i >> 2;
    }
}
