/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class EqualToIndex {
    @Positive
  static int[] a = {0};

    @Positive
  public static void equalToUpper(@LTLengthOf("a") int m, @LTEqLengthOf("a") int r) {
    @Positive
    if (r == m) {
    @Positive
      @LTLengthOf("a") int j = r;
    @Positive
    }
    @Positive
  }
    @Positive
}
