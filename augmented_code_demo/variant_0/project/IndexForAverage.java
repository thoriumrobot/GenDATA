// test case for issue 86: https://github.com/kelloggm/checker-framework/issues/86

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class IndexForAverage {

    @Positive
  public static void bug(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @IndexFor("a") int k = (i + j) / 2;
    @Positive
  }

    @Positive
  public static void bug2(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @LTLengthOf("a") int k = ((i - 1) + j) / 2;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int h = ((i + 1) + j) / 2;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
