    @Positive
import org.checkerframework.checker.index.qual.SameLen;

    @Positive
public class SameLenLUBStrangeness {
    @Positive
  void test(int[] a, boolean cond) {
    @Positive
    int[] b;
    @Positive
    if (cond) {
    @Positive
      b = a;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    int @SameLen({"a", "b"}) [] c = a;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
