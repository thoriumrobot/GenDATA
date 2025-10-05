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
public class RefinementLT {

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards less than */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 < a) {
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 < j) {
    @Positive
      @Positive int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 < s) {
    @Positive
      @Positive int t = s;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_forwards(int a, int j, int s) {
    /** forwards less than */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (a < -1) {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j < 0) {
      // :: error: (assignment)
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
    @Positive
      @NonNegative int l = j;
    @Positive
    }

    @Positive
    if (s < 1) {
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
