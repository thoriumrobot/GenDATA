    @Positive
import org.checkerframework.checker.index.qual.LessThan;

    @Positive
public class LessThanZeroArrayLength {
    @Positive
  void test(int[] a) {
    @Positive
    foo(0, a.length);
    @Positive
  }

    @Positive
  void foo(@LessThan("#2 + 1") int x, int y) {}
    @Positive
}

// CFWR semantic augmentation - variant 0
