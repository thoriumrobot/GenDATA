    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.LowerBoundUnknown;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class IntroRules {

    @Positive
  void test() {
    @Positive
    @Positive int a = 10;
    @Positive
    @NonNegative int b = 9;
    @Positive
    @GTENegativeOne int c = 8;
    @Positive
    @LowerBoundUnknown int d = 7;

    // :: error: (assignment)
    @Positive
    @Positive int e = 0;
    // :: error: (assignment)
    @Positive
    @Positive int f = -1;
    // :: error: (assignment)
    @Positive
    @Positive int g = -6;

    @Positive
    @NonNegative int h = 0;
    @Positive
    @GTENegativeOne int i = 0;
    @Positive
    @LowerBoundUnknown int j = 0;
    // :: error: (assignment)
    @Positive
    @NonNegative int k = -1;
    // :: error: (assignment)
    @Positive
    @NonNegative int l = -4;

    @Positive
    @GTENegativeOne int m = -1;
    @Positive
    @LowerBoundUnknown int n = -1;
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int o = -9;
    @Positive
  }
    @Positive
}
// a comment

// CFWR semantic augmentation - variant 0
