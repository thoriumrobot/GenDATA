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
public class TransferSub {

    @Positive
  void test() {
    // zero, one, and two
    @Positive
    int a = 1;

    @Positive
    @NonNegative int b = a - 1;
    // :: error: (assignment)
    @Positive
    @Positive int c = a - 1;
    @Positive
    @GTENegativeOne int d = a - 2;

    // :: error: (assignment)
    @Positive
    @NonNegative int e = a - 2;

    @Positive
    @GTENegativeOne int f = b - 1;
    // :: error: (assignment)
    @Positive
    @NonNegative int g = b - 1;

    // :: error: (assignment)
    @Positive
    @GTENegativeOne int h = f - 1;

    @Positive
    @GTENegativeOne int i = f;
    @Positive
    @NonNegative int j = b;
    @Positive
    @Positive int k = a;

    // :: error: (assignment)
    @Positive
    @Positive int l = j;
    // :: error: (assignment)
    @Positive
    @NonNegative int m = i;

    // :: error: (assignment)
    @Positive
    @Positive int n = a - k;
    // this would be an error if the values of b and j (both zero) weren't known at compile time
    @Positive
    @NonNegative int o = b - j;
    /* i and d both have compile time value -1, so this is legal.
    @Positive
    The general case of GTEN1 - GTEN1 is not, though. */
    @Positive
    @GTENegativeOne int p = i - d;

    // decrements

    // :: error: (unary.decrement) :: error: (assignment)
    @Positive
    @Positive int q = --k; // k = 0

    // :: error: (unary.decrement)
    @Positive
    @NonNegative int r = k--; // after this k = -1

    @Positive
    int k1 = 0;
    @Positive
    @NonNegative int s = k1--;

    // :: error: (assignment)
    @Positive
    @NonNegative int s1 = k1;

    // transferred to SimpleTransferSub.java
    // this section is failing due to CF bug
    // int k2 = 0;
    // // :: error: (assignment)
    // @Positive int s2 = k2--;

    @Positive
    k1 = 1;
    @Positive
    @NonNegative int t = --k1;

    @Positive
    k1 = 1;
    // :: error: (assignment)
    @Positive
    @Positive int t1 = --k1;

    @Positive
    int u1 = -1;
    @Positive
    @GTENegativeOne int x = u1--;
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int x1 = u1;
    @Positive
  }
    @Positive
}
// a comment
