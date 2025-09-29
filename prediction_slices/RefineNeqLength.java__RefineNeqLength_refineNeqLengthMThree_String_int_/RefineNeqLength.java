import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LTOMLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.common.value.qual.IntVal;

public class RefineNeqLength {

    @LTLengthOf(value = "#1", offset = "3")
    int refineNeqLengthMThree(int[] array, @NonNegative @LTLengthOf(value = "#1", offset = "2") int i) {
        if (i != array.length - 3) {
            return i;
        }
        return i;
    }
}
