import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LTOMLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.common.value.qual.IntVal;

public class RefineNeqLength {

    void refineNeqLengthMTwo(int[] array, @NonNegative @LTOMLengthOf("#1") int i) {
        if (i != array.length - 2) {
            refineNeqLengthMThree(array, i);
        }
        if (i != array.length - 1) {
            refineNeqLengthMThree(array, i);
        }
    }
}
