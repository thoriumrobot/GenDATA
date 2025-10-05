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
public class TransferAdd {

    @Positive
  void test() {

    // adding zero and one and two

    @Positive
    int a = -1;

    @Positive
    @Positive int a1 = a + 2;

    @Positive
    @NonNegative int b = a + 1;
    @Positive
    @NonNegative int c = 1 + a;

    @Positive
    @GTENegativeOne int d = a + 0;
    @Positive
    @GTENegativeOne int e = 0 + a;

    // :: error: (assignment)
    @Positive
    @Positive int f = a + 1;

    @Positive
    @NonNegative int g = b + 0;

    @Positive
    @Positive int h = b + 1;

    @Positive
    @Positive int i = h + 1;
    @Positive
    @Positive int j = h + 0;

    // adding values

    @Positive
    @Positive int k = i + j;
    // :: error: (assignment)
    @Positive
    @Positive int l = b + c;
    // :: error: (assignment)
    @Positive
    @Positive int m = d + c;
    // :: error: (assignment)
    @Positive
    @Positive int n = d + e;

    @Positive
    @Positive int o = h + g;
    // :: error: (assignment)
    @Positive
    @Positive int p = h + d;

    @Positive
    @NonNegative int q = b + c;
    // :: error: (assignment)
    @Positive
    @NonNegative int r = q + d;

    @Positive
    @NonNegative int s = k + d;
    @Positive
    @GTENegativeOne int t = s + d;

    // increments

    // :: error: (assignment)
    @Positive
    @Positive int u = b++;

    @Positive
    @Positive int u1 = b;

    @Positive
    @Positive int v = ++c;

    @Positive
    @Positive int v1 = c;

    @Positive
    int n1p1 = -1, n1p2 = -1;

    @Positive
    @NonNegative int w = ++n1p1;

    @Positive
    @NonNegative int w1 = n1p1;

    // :: error: (assignment)
    @Positive
    @Positive int w2 = n1p1;
    // :: error: (assignment)
    @Positive
    @Positive int w3 = n1p1++;

    // :: error: (assignment)
    @Positive
    @NonNegative int x = n1p2++;

    @Positive
    @NonNegative int x1 = n1p2;

    // :: error: (assignment)
    @Positive
    @Positive int y = ++d;
    // :: error: (assignment)
    @Positive
    @Positive int z = e++;
    @Positive
  }
    @Positive
}
// a comment
