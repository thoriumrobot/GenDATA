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
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment
