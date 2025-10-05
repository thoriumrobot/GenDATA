/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// @skip-test until fixed.

    @Positive
public class CompareBySubtraction {
    @Positive
  public int compare(int[] a1, int[] a2) {
    @Positive
    if (a1 == a2) {
    @Positive
      return 0;
    @Positive
    }
    @Positive
    int tmp;
    @Positive
    tmp = a1.length - a2.length;
    @Positive
    if (tmp != 0) {
    @Positive
      return tmp;
    @Positive
    }
    @Positive
    int i = 0;
        while (i < a1.length) {
            @Positive
      if (a1[i] != a2[i]) {
    @Positive
        return ((a1[i] > a2[i]) ? 1 : -1);
    @Positive
      }
    @Positive
            i++;
        }
    @Positive
    return 0;
    @Positive
  }
    @Positive
}
