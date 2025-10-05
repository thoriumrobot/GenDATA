/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for Issue 185:
// https://github.com/typetools/kelloggm/issues/185

    @Positive
import java.util.BitSet;
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;

    @Positive
public class BitSetLowerBound {

    @Positive
  private void m(BitSet b) {
    @Positive
    b.set(b.nextClearBit(0));
    // next set bit does not have to exist
    // :: error: (argument)
    @Positive
    b.clear(b.nextSetBit(0));
    @Positive
    @GTENegativeOne int i = b.nextSetBit(0);

    @Positive
    @GTENegativeOne int j = b.previousClearBit(-1);
    @Positive
    @GTENegativeOne int k = b.previousSetBit(-1);
    @Positive
  }
    @Positive
}
