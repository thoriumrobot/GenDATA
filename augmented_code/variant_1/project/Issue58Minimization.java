/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.index.qual.SameLen;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class Issue58Minimization {

    @Positive
  void test(@GTENegativeOne int x) {
    @Positive
    int z;
    @Positive
    if ((z = x) != -1) {
    @Positive
      @NonNegative int y = z;
    @Positive
    }
    @Positive
    if ((z = x) != 1) {
      // :: error: (assignment)
    @Positive
      @NonNegative int y = z;
    @Positive
    }
    @Positive
  }

    @Positive
  void test2(@NonNegative int x) {
    @Positive
    int z;
    @Positive
    if ((z = x) != 0) {
    @Positive
      @Positive int y = z;
    @Positive
    }
    @Positive
    if ((z = x) == 0) {
      // do nothing
    @Positive
      int y = 5;
    @Positive
    } else {
    @Positive
      @Positive int y = x;
    @Positive
    }
    @Positive
  }

    @Positive
  void ubc_test(int[] a, @LTEqLengthOf("#1") int x) {
    @Positive
    int z;
    @Positive
    if ((z = x) != a.length) {
    @Positive
      @LTLengthOf("a") int y = z;
    @Positive
    }
    @Positive
  }

    @Positive
  void samelen_test(int[] a, int[] c) {
    @Positive
    int[] b;
    @Positive
    if ((b = a) == c) {
    @Positive
      int @SameLen({"a", "b", "c"}) [] d = b;
    @Positive
    }
    @Positive
  }

    @Positive
  void minlen_test(int[] a, int @MinLen(1) [] c) {
    @Positive
    int[] b;
    @Positive
    if ((b = a) == c) {
    @Positive
      int @MinLen(1) [] d = b;
    @Positive
    }
    @Positive
  }

    @Positive
  void minlen_test2(int[] a, int x) {
    @Positive
    int one = 1;
    @Positive
    if ((x = one) == a.length) {
    @Positive
      int @MinLen(1) [] b = a;
    @Positive
    }
    @Positive
  }
    @Positive
}
