    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class TransferMod {

    @Positive
  void test() {
    @Positive
    int aa = -100;
    @Positive
    int a = -1;
    @Positive
    int b = 0;
    @Positive
    int c = 1;
    @Positive
    int d = 2;

    @Positive
    @Positive int e = 5 % 3;
    @Positive
    @NonNegative int f = -100 % 1;

    @Positive
    @NonNegative int g = aa % -1;
    @Positive
    @NonNegative int h = aa % 1;
    @Positive
    @NonNegative int i = d % -1;
    @Positive
    @NonNegative int j = d % 1;

    @Positive
    @NonNegative int k = d % c;
    @Positive
    @NonNegative int l = b % c;
    @Positive
    @NonNegative int m = c % d;

    @Positive
    @NonNegative int n = c % a;
    @Positive
    @NonNegative int o = b % a;

    @Positive
    @GTENegativeOne int p = a % a;
    @Positive
    @GTENegativeOne int q = a % d;
    @Positive
    @GTENegativeOne int r = a % c;
    @Positive
  }
    @Positive
}
// a comment

// CFWR semantic augmentation - variant 0
