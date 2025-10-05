    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class TransferDivide {

    @Positive
  void test() {
    @Positive
    int a = -1;
    @Positive
    int b = 0;
    @Positive
    int c = 1;
    @Positive
    int d = 2;

    /** literals */
    @Positive
    @Positive int e = -1 / -1;

    /** 0 / * -> NN */
    @Positive
    @NonNegative int f = 0 / a;
    @Positive
    @NonNegative int g = 0 / d;

    /** * / 1 -> * */
    @Positive
    @GTENegativeOne int h = a / 1;
    @Positive
    @NonNegative int i = b / 1;
    @Positive
    @Positive int j = c / 1;
    @Positive
    @Positive int k = d / 1;

    /** pos / pos -> nn */
    @Positive
    @NonNegative int l = d / c;
    @Positive
    @NonNegative int m = c / d;
    // :: error: (assignment)
    @Positive
    @Positive int n = c / d;

    /** nn / pos -> nn */
    @Positive
    @NonNegative int o = b / c;
    // :: error: (assignment)
    @Positive
    @Positive int p = b / d;

    /** pos / nn -> nn */
    @Positive
    @NonNegative int q = d / l;
    // :: error: (assignment)
    @Positive
    @Positive int r = c / l;

    /** nn / nn -> nn */
    @Positive
    @NonNegative int s = b / q;
    // :: error: (assignment)
    @Positive
    @Positive int t = b / q;

    /** n1p / pos -> n1p */
    @Positive
    @GTENegativeOne int u = a / d;
    @Positive
    @GTENegativeOne int v = a / c;
    // :: error: (assignment)
    @Positive
    @NonNegative int w = a / c;

    /** n1p / nn -> n1p */
    @Positive
    @GTENegativeOne int x = a / l;
    @Positive
  }

    @Positive
  void testDivideByTwo(@NonNegative int x) {
    @Positive
    @NonNegative int y = x / 2;
    @Positive
  }
    @Positive
}
// a comment

// CFWR semantic augmentation - variant 0
