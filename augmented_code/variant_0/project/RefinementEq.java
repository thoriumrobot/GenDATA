/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementEq {

    @Positive
  void test_equal(int a, int j, int s) {

    @Positive
    if (-1 != a) {
            // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
        } else {
            @Positive
      @GTENegativeOne int b = a;
    @Positive
        }

    @Positive
    if (0 != j) {
            // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
        } else {
            @Positive
      @NonNegative int k = j;
    @Positive
        }

    @Positive
    if (1 != s) {
            // :: error: (assignment)
    @Positive
      @Positive int u = s;
    @Positive
        } else {
            @Positive
      @Positive int t = s;
    @Positive
        }
    @Positive
  }
    @Positive
}
// a comment
