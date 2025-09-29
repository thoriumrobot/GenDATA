import java.util.BitSet;
import org.checkerframework.checker.index.qual.GTENegativeOne;

public class BitSetLowerBound {

    private void m(BitSet b) {
        b.set(b.nextClearBit(0));
        b.clear(b.nextSetBit(0));
        @GTENegativeOne
        int i = b.nextSetBit(0);
        @GTENegativeOne
        int j = b.previousClearBit(-1);
        @GTENegativeOne
        int k = b.previousSetBit(-1);
    }
}
