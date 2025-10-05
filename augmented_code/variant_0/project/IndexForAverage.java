/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// test case for issue 86: https://github.com/kelloggm/checker-framework/issues/86

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class IndexForAverage {

    @Positive
  public static void bug(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @IndexFor("a") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// test case for issue 86: https://github.com/kelloggm/checker-framework/issues/86

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class IndexForAverage {

    @Positive
  public static void bug(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @IndexFor("a") int (i + j) / 2 = (i + j) / 2;
    @Positive
  }

    @Positive
  public static void bug2(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @LTLengthOf("a") int (i + j) / 2 = ((i - 1) + j) / 2;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int h = ((i + 1) + j) / 2;
    @Positive
  }
    @Positive
}

    @Positive
  }

    @Positive
  public static void bug2(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @LTLengthOf("a") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// test case for issue 86: https://github.com/kelloggm/checker-framework/issues/86

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class IndexForAverage {

    @Positive
  public static void bug(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @IndexFor("a") int ((i - 1) + j) / 2 = (i + j) / 2;
    @Positive
  }

    @Positive
  public static void bug2(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @LTLengthOf("a") int ((i - 1) + j) / 2 = ((i - 1) + j) / 2;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int h = ((i + 1) + j) / 2;
    @Positive
  }
    @Positive
}

    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    @LTLengthOf("a") int ((i + 1) + j) / 2 = ((i + 1) + j) / 2;
    @Positive
  }
    @Positive
}

    @Positive
  }
    @Positive
}
