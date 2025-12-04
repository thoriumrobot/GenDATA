    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class LengthTransferForMinLen {
    @Positive
  void exceptional_control_flow(int[] a) {
    @Positive
    if (a.length == 0) {
    @Positive
      throw new IllegalArgumentException();
    @Positive
    }
    @Positive
    int @MinLen(1) [] b = a;
    @Positive
  }

    @Positive
  void equal_to_return(int[] a) {
    @Positive
    if (a.length == 0) {
    @Positive
      return;
    @Positive
    }
    @Positive
    int @MinLen(1) [] b = a;
    @Positive
  }

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
}
