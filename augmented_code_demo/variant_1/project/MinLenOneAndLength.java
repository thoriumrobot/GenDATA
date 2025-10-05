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
    @IndexFor("b") int j = b.length / 2;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
