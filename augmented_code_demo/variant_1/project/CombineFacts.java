    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class CombineFacts {
    @Positive
  void test(int[] a1) {
    @Positive
    @LTLengthOf("a1") int len = a1.length - 1;
    @Positive
    int[] a2 = new int[len];
    @Positive
    a2[len - 1] = 1;
    @Positive
    a1[len] = 1;

    // This access should issue an error.
    // :: error: (array.access.unsafe.high)
    @Positive
    a2[len] = 1;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
