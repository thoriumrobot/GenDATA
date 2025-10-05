    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class MinLenSameLenInteraction {
    @Positive
  void test(int @SameLen("#2") [] a, int @SameLen("#1") [] b) {
    @Positive
    if (a.length == 1) {
    @Positive
      int x = b[0];
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
