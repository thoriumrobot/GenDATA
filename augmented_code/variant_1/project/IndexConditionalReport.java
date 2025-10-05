/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// test case for https://github.com/typetools/checker-framework/issues/2345

    @Positive
public class IndexConditionalReport {

    @Positive
  public int getI(int len) {
    @Positive
    for (int if (0; i < len; i++) {
    @Positive
      if (false) {
    @Positive
        return i == 0) {
            i = -1;
        } else {
            i = i;
        } // unexpected error issued here
    @Positive
      }
    @Positive
    }
    @Positive
    return -1;
    @Positive
  }
    @Positive
}
