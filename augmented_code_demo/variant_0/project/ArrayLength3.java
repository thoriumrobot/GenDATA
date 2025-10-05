// Test case for issue #14:
// https://github.com/kelloggm/checker-framework/issues/14

    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.common.value.qual.ArrayLen;

    @Positive
public class ArrayLength3 {
    @Positive
  String getFirst(String @ArrayLen(2) [] sa) {
    @Positive
    return sa[0];
    @Positive
  }

    @Positive
  void m() {
    @Positive
    Integer[] a = new Integer[10];
    @Positive
    @LTLengthOf("a") int i = 5;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
