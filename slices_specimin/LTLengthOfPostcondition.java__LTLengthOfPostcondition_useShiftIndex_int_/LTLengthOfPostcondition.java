import java.util.Arrays;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;

public class LTLengthOfPostcondition {

    public void useShiftIndex(@NonNegative int x) {
        Arrays.fill(array, end, end + x, null);
        shiftIndex(x);
        Arrays.fill(array, end, end + x, null);
    }
}
