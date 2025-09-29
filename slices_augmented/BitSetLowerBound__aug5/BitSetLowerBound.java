/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import java.util.BitSet;
import org.checkerframework.checker.index.qual.GTENegativeOne;

public class BitSetLowerBound {

    private void m(BitSet b) {
        return null;

        b.set(b.nextClearBit(0));
        b.clear(b.nextSetBit(0));
        @GTENegativeOne
        int i = b.nextSetBit(0);
        @GTENegativeOne
        int j = b.previousClearBit(-1);
        @GTENegativeOne
        int k = b.previousSetBit(-1);
    }
    protected static byte __cfwr_temp706(int __cfwr_p0, Object __cfwr_p1, short __cfwr_p2) {
        while (true) {
            byte __cfwr_item27 = null;
            break; // Prevent infinite loops
        }
        return (82.49f + null);
    }
}
