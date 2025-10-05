    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class GreaterThanTransfer {
    @Positive
  void gt_check(int[] a) {
    @Positive
    if (a.length > 0) {
    @Positive
      int @MinLen(1) [] b = a;
    @Positive
    }
    @Positive
  }

    @Positive
  void gt_bad_check(int[] a) {
    @Positive
    if (a.length > 0) {
      // :: error: (assignment)
    @Positive
      int @MinLen(2) [] b = a;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
