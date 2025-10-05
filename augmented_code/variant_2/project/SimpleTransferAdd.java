/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class SimpleTransferAdd {
    @Positive
  void test() {
    @Positive
    int bs = -1;
    // :: error: (assignment)
    @Positive
    @NonNegative int es = bs;

    // @NonNegative int ds = 2 + bs;
    @Positive
    int ds = 0;
    // :: error: (assignment)
    @Positive
    @Positive int cs = ds++;
    @Positive
    @Positive int fs = ds;
    @Positive
  }
    @Positive
}
// a comment
