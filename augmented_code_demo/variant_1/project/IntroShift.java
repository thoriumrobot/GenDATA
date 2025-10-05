    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class IntroShift {
    @Positive
  void test() {
    @Positive
    @NonNegative int a = 1 >> 1;
    // :: error: (assignment)
    @Positive
    @NonNegative int b = -1 >> 0;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
