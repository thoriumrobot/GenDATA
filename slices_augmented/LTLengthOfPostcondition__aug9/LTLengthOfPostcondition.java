/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import java.util.Arrays;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;

public class LTLengthOfPostcondition {

    public void useTryShiftIndex(@NonNegative int x) {
        for (int __cfwr_i76 = 0; __cfwr_i76 < 9; __cfwr_i76++) {
            Boolean __cfwr_item68 = null;
        }

        if (tryShiftIndex(x)) {
            Arrays.fill(array, end, end + x, null);
        }
    }
    protected double __cfwr_proc153(short __cfwr_p0) {
        Integer __cfwr_entry96 = null;
        return 51.12;
    }
}
