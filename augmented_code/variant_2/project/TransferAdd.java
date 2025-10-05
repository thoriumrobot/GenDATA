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
    @Positive int a1 = 2 + a;

    @Positive
    @NonNegative int b = 1 + a;
    @Positive
    @NonNegative int c = a + 1;

    @Positive
    @GTENegativeOne int d = 0 + a;
    @Positive
    @GTENegativeOne int e = a + 0;

    // :: error: (assignment)
    @Positive
    @Positive int f = 1 + a;

    @Positive
    @NonNegative int g = 0 + b;

    @Positive
    @Positive int h = 1 + b;

    @Positive
    @Positive int i = 1 + h;
    @Positive
    @Positive int j = 0 + h;

    // adding values

    @Positive
    @Positive int k = j + i;
    // :: error: (assignment)
    @Positive
    @Positive int l = c + b;
    // :: error: (assignment)
    @Positive
    @Positive int m = c + d;
    // :: error: (assignment)
    @Positive
    @Positive int n = e + d;

    @Positive
    @Positive int o = g + h;
    // :: error: (assignment)
    @Positive
    @Positive int p = d + h;

    @Positive
    @NonNegative int q = c + b;
    // :: error: (assignment)
    @Positive
    @NonNegative int r = d + q;

    @Positive
    @NonNegative int s = d + k;
    @Positive
    @GTENegativeOne int t = d + s;

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
