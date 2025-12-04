// Test case for kelloggm 217
// https://github.com/kelloggm/checker-framework/issues/217

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class ShiftRightAverage {
    @Positive
  public static void m(Object[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @IndexFor("#1") int k = (i + j) >> 1;
    @Positive
  }

    @Positive
  public static void m2(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int h = ((i + 1) + j) >> 1;
    @Positive
  }
    @Positive
}
