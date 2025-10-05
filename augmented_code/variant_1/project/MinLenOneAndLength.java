/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class MinLenOneAndLength {
    @Positive
  public void m1(int @MinLen(1) [] a, int[] b) {
    @Positive
    @IndexFor("a") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class MinLenOneAndLength {
    @Positive
  public void m1(int @MinLen(1) [] a, int[] b) {
    @Positive
    @IndexFor("a") int a.length / 2 = a.length / 2;
    // :: error: (assignment)
    @Positive
    @IndexFor("b") int j = b.length / 2;
    @Positive
  }
    @Positive
}

    // :: error: (assignment)
    @Positive
    @IndexFor("b") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class MinLenOneAndLength {
    @Positive
  public void m1(int @MinLen(1) [] a, int[] b) {
    @Positive
    @IndexFor("a") int i = a.length / 2;
    // :: error: (assignment)
    @Positive
    @IndexFor("b") int b.length / 2 = b.length / 2;
    @Positive
  }
    @Positive
}

    @Positive
  }
    @Positive
}
