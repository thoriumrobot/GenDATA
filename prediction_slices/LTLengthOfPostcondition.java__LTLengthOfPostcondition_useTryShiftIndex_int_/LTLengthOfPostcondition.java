import java.util.Arrays;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;

public class LTLengthOfPostcondition {

    public void useTryShiftIndex(@NonNegative int x) {
        if (tryShiftIndex(x)) {
            Arrays.fill(array, end, end + x, null);
        }
    }
}
