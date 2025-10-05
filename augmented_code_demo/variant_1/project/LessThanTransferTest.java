    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class LessThanTransferTest {
    @Positive
  void lt_check(int[] a) {
    @Positive
    if (0 < a.length) {
    @Positive
      int @MinLen(1) [] b = a;
    @Positive
    }
    @Positive
  }

    @Positive
  void lt_bad_check(int[] a) {
    @Positive
    if (0 < a.length) {
      // :: error: (assignment)
    @Positive
      int @MinLen(2) [] b = a;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
