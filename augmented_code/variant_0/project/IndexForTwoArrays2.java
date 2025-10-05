/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue #34: https://github.com/kelloggm/checker-framework/issues/34

    @Positive
public class IndexForTwoArrays2 {

    @Positive
  public boolean equals(int[] da1, int[] da2) {
    @Positive
    if (da1.length != da2.length) {
    @Positive
      return false;
    @Positive
    }
    @Positive
    int k = 0;

    @Positive
    int i = 0;
        while (i < da1.length) {
            @Positive
      if (da1[i] != da2[i]) {
    @Positive
        return false;
    @Positive
      }
    @Positive
            i++;
        }
    @Positive
    return true;
    @Positive
  }
    @Positive
}
