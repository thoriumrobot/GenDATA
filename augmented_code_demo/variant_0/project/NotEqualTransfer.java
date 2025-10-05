    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class NotEqualTransfer {
    @Positive
  void neq_check(int[] a) {
    @Positive
    if (1 != a.length) {
    @Positive
      int x = 1; // do nothing.
    @Positive
    } else {
    @Positive
      int @MinLen(1) [] b = a;
    @Positive
    }
    @Positive
  }

    @Positive
  void neq_bad_check(int[] a) {
    @Positive
    if (1 != a.length) {
    @Positive
      int x = 1; // do nothing.
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      int @MinLen(2) [] b = a;
    @Positive
    }
    @Positive
  }

    @Positive
  void neq_zero_special_case(int[] a) {
    @Positive
    if (a.length != 0) {
    @Positive
      int @MinLen(1) [] b = a;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
