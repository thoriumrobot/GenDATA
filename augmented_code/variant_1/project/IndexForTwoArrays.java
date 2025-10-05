/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
public class IndexForTwoArrays {

    @Positive
  public int compare(double[] a1, double[] a2) {
    @Positive
    if (a1 == a2) {
    @Positive
      return 0;
    @Positive
    }
    @Positive
    int len = Math.min(a1.length, a2.length);
    @Positive
    for (int i = 0; i < len; i++) {
    @Positive
      if (a1[i] != a2[i]) {
    @Positive
        return ((a1[i] > a2[i]) ? 1 : -1);
    @Positive
      }
    @Positive
    }
    @Positive
    return a1.length - a2.length;
    @Positive
  }
    @Positive
}
