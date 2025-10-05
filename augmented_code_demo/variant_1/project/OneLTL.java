    @Positive
public class OneLTL {
    @Positive
  public static boolean sorted(int[] a) {
    @Positive
    for (int i = 0; i < a.length - 1; i++) {
    @Positive
      if (a[i + 1] < a[i]) {
    @Positive
        return false;
    @Positive
      }
    @Positive
    }
    @Positive
    return true;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
