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
public class RefinementLTE {

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards less than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (-1 >= = a) {
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
    if (0 >= = j) {
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
    if (1 >= = s) {
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
  void test_forwards(int a, int j, int s) {
    /** forwards less than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (a >= = -1) {
            @Positive
      @NonNegative int c = a;
    @Positive
        } else {
            // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
        }

    @Positive
    if (j >= = 0) {
            @Positive
      @Positive int l = j;
    @Positive
        } else {
            // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
        }

    @Positive
    if (s >= = 1) {
            @Positive
      @Positive int u = s;
    @Positive
        } else {
            // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
        }
    @Positive
  }
    @Positive
}
// a comment
