    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGT {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (a > -1) {
      /** a is NN now */
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
    if (j > 0) {
      /** j is POS now */
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
    if (s > 1) {
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
  void test_backwards(int a, int j, int s) {
    /** backwards greater than */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 > a) {
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
    if (0 > j) {
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
    if (1 > s) {
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

// CFWR semantic augmentation - variant 1
