    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.LowerBoundUnknown;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class LBCSubtyping {

    @Positive
  void foo() {

    @Positive
    @GTENegativeOne int i = -1;

    @Positive
    @LowerBoundUnknown int j = i;

    @Positive
    int k = -4;

    // not this one though
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int l = k;

    @Positive
    @NonNegative int n = 0;

    @Positive
    @Positive int a = 1;

    // check that everything is aboveboard
    @Positive
    j = a;
    @Positive
    j = n;
    @Positive
    l = n;
    @Positive
    n = a;

    // error cases

    // :: error: (assignment)
    @Positive
    @NonNegative int p = i;
    // :: error: (assignment)
    @Positive
    @Positive int b = i;

    // :: error: (assignment)
    @Positive
    @NonNegative int r = k;
    // :: error: (assignment)
    @Positive
    @Positive int c = k;

    // :: error: (assignment)
    @Positive
    @Positive int d = r;
    @Positive
  }
    @Positive
}
// a comment

// CFWR semantic augmentation - variant 1
