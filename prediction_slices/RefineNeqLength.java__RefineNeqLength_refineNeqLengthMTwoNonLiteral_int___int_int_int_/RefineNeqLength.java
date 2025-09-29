import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LTOMLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.common.value.qual.IntVal;

public class RefineNeqLength {

    void refineNeqLengthMTwoNonLiteral(int[] array, @NonNegative @LTOMLengthOf("#1") int i, @IntVal(3) int c3, @IntVal({ 2, 3 }) int c23) {
        if (i != array.length - (5 - c3)) {
            refineNeqLengthMThree(array, i);
        }
        if (i != array.length - c23) {
            refineNeqLengthMThree(array, i);
        }
    }
}
