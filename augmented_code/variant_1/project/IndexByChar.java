/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
public class IndexByChar {
    @Positive
  public int m(char c) {
    @Positive
    int[] i = new int[128];
    @Positive
    if (c >= 128) {
            @Positive
      return -1;
    @Positive
        } else {
            @Positive
      return i[c];
    @Positive
        }
    @Positive
  }
    @Positive
}
