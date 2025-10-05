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
    if (-1 == a) {
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
    if (0 == j) {
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
    if (1 == s) {
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
}
// a comment

// CFWR semantic augmentation - variant 1
