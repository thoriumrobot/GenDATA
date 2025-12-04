    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class GreaterThanOrEqualTransfer {
    @Positive
  void gte_check(int[] a) {
    @Positive
    if (a.length >= 1) {
    @Positive
      int @MinLen(1) [] b = a;
    @Positive
    }
    @Positive
  }

    @Positive
  void gte_bad_check(int[] a) {
    @Positive
    if (a.length >= 1) {
      // :: error: (assignment)
    @Positive
      int @MinLen(2) [] b = a;
    @Positive
    }
    @Positive
  }
    @Positive
}
