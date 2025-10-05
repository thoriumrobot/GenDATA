    @Positive
public class LengthTransfer {
    @Positive
  void exceptional_control_flow(int[] a) {
    @Positive
    if (a.length == 0) {
    @Positive
      throw new IllegalArgumentException();
    @Positive
    }
    @Positive
    int i = a[0];
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
    int i = a[0];
    @Positive
  }

    @Positive
  void gt_check(int[] a) {
    @Positive
    if (a.length > 0) {
    @Positive
      int i = a[0];
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
