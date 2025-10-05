    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class Issue20 {
  // An issue with LUB that results in losing information when unifying.
    @Positive
  int[] a, b;

    @Positive
  void test(@LTLengthOf("a") int i, @LTEqLengthOf({"a", "b"}) int j, boolean flag) {
    @Positive
    @LTEqLengthOf("a") int k = flag ? i : j;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
