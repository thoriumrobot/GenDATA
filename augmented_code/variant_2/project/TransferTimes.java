/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class TransferTimes {

    @Positive
  void test() {
    @Positive
    int a = 1;
    @Positive
    @Positive int b = a * 1;
    @Positive
    @Positive int c = 1 * a;
    @Positive
    @NonNegative int d = 0 * a;
    // :: error: (assignment)
    @Positive
    @NonNegative int e = -1 * a;

    @Positive
    int g = -1;
    @Positive
    @NonNegative int h = g * 0;
    // :: error: (assignment)
    @Positive
    @Positive int i = g * 0;
    // :: error: (assignment)
    @Positive
    @Positive int j = g * a;

    @Positive
    int k = 0;
    @Positive
    int l = 1;
    @Positive
    @Positive int m = a * l;
    @Positive
    @NonNegative int n = k * l;
    @Positive
    @NonNegative int o = k * k;
    @Positive
  }
    @Positive
}
// a comment
