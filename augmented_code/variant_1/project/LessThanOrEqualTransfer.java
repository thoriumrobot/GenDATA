/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class LessThanOrEqualTransfer {
    @Positive
  void lte_check(int[] a) {
    @Positive
    if (1 <= a.length) {
    @Positive
      int @MinLen(1) [] b = a;
    @Positive
    }
    @Positive
  }

    @Positive
  void lte_bad_check(int[] a) {
    @Positive
    if (1 <= a.length) {
      // :: error: (assignment)
    @Positive
      int @MinLen(2) [] b = a;
    @Positive
    }
    @Positive
  }
    @Positive
}
