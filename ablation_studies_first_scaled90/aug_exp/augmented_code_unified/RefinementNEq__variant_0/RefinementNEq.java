    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementNEq {

    @Positive
  void test_not_equal(int a, int j, int s) {

    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 != a) {
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
    if (0 != j) {
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
    if (1 != s) {
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
