    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class TypeArrayLengthWithSameLen {
    @Positive
  void test(int @SameLen("#2") [] a, int @SameLen("#1") [] b, int[] c) {
    @Positive
    if (a.length == c.length) {
    @Positive
      @LTEqLengthOf({"a", "b", "c"}) int x = b.length;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
