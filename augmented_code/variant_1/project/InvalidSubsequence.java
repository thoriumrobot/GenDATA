/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.HasSubsequence;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;

    @Positive
public class InvalidSubsequence {
  // :: error: (flowexpr.parse.error) :: error: (not.final)
    @Positive
  int[] a;

  // :: error: (flowexpr.parse.error) :: error: (not.final)
    @Positive
  int[] b;

  // :: error: (flowexpr.parse.error) :: error: (not.final)
    @Positive
  int[] c;

  // :: error: (not.final)
    @Positive
  int[] e;

    @Positive
  @IndexFor("a") @LessThan("to") int from;

    @Positive
  @IndexOrHigh("a") int to;

    @Positive
  void assignA(int[] d) {
    // :: error: (to.not.ltel)
    @Positive
    a = d;
    @Positive
  }

    @Positive
  void assignB(int[] d) {
    // :: error: (from.gt.to) :: error: (from.not.nonnegative) :: error: (to.not.ltel)
    @Positive
    b = d;
    @Positive
  }

    @Positive
  void assignC(int[] d) {
    // :: error: (from.gt.to) :: error: (to.not.ltel)
    @Positive
    c = d;
    @Positive
  }

    @Positive
  void assignE(int[] d) {
    // :: error: (from.gt.to) :: error: (to.not.ltel)
    @Positive
    e = d;
    @Positive
  }
    @Positive
}
