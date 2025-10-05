    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class PlusPlusBug {
    @Positive
  int[] array = {};

    @Positive
  void test(@LTLengthOf("array") int x) {
    // :: error: (unary.increment)
    @Positive
    x++;
    // :: error: (unary.increment)
    @Positive
    ++x;
    // :: error: (assignment)
    @Positive
    x = x + 1;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
