// Test case for kelloggm#216
// https://github.com/kelloggm/checker-framework/issues/216

    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class NegativeIndex {
    @Positive
  void m(int[] a) {
    @Positive
    int x = a[-100];
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
