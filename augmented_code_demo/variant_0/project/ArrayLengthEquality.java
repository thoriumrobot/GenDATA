    @Positive
import org.checkerframework.checker.index.qual.SameLen;

    @Positive
public class ArrayLengthEquality {
    @Positive
  void test(int[] a, int[] b) {
    @Positive
    if (a.length == b.length) {
    @Positive
      int @SameLen({"a", "b"}) [] c = a;
    @Positive
      int @SameLen({"a", "b"}) [] d = b;
    @Positive
    }
    @Positive
    if (a.length != b.length) {
      // Do nothing.
    @Positive
      int dead_variable = 0;
    @Positive
    } else {
    @Positive
      int @SameLen({"a", "b"}) [] e = a;
    @Positive
      int @SameLen({"a", "b"}) [] f = b;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
