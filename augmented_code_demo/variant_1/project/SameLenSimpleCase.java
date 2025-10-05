    @Positive
public class SameLenSimpleCase {
    @Positive
  public int compare(int[] a1, int[] a2) {
    @Positive
    if (a1.length != a2.length) {
    @Positive
      return a1.length - a2.length;
    @Positive
    }
    @Positive
    for (int i = 0; i < a1.length; i++) {
    @Positive
      if (a1[i] != a2[i]) {
    @Positive
        return ((a1[i] > a2[i]) ? 1 : -1);
    @Positive
      }
    @Positive
    }
    @Positive
    return 0;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
