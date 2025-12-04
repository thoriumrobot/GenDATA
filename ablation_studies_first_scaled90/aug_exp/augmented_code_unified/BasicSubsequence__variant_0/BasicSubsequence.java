    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class BasicSubsequence {
  // :: error: (not.final)
    @Positive
  int[] b;

    @Positive
  int x;
    @Positive
  int y;

    @Positive
  void test2(@NonNegative @LessThan("y + 1") int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test4(@NonNegative int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (from.gt.to) :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test5(@GTENegativeOne @LessThan("y + 1") int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (from.not.nonnegative) :: error: (to.not.ltel)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test6(@GTENegativeOne int x1, int[] a) {
    @Positive
    x = x1;
    // :: error: (from.not.nonnegative) :: error: (to.not.ltel) :: error: (from.gt.to)
    @Positive
    b = a;
    @Positive
  }

    @Positive
  void test7(@IndexFor("this") @LessThan("y") int x1, @IndexOrHigh("this") int y1, int[] a) {
    @Positive
    x = x1;
    @Positive
    y = y1;
    // :: warning: (which.subsequence)
    @Positive
    b = a;
    @Positive
  }
    @Positive
}
