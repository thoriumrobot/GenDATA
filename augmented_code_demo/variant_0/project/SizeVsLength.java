// test case for issue 91: https://github.com/kelloggm/checker-framework/issues/91

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SizeVsLength {

    @Positive
  public int[] getArray(@NonNegative int size) {
    @Positive
    int[] values = new int[size];
    @Positive
    for (int i = 0; i < size; i++) {
    @Positive
      values[i] = 22;
    @Positive
    }
    @Positive
    return values;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
