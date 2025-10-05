// test case for https://github.com/typetools/checker-framework/issues/2345

    @Positive
public class IndexConditionalReport {

    @Positive
  public int getI(int len) {
    @Positive
    for (int i = 0; i < len; i++) {
    @Positive
      if (false) {
    @Positive
        return i == 0 ? -1 : i; // unexpected error issued here
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

// CFWR semantic augmentation - variant 1
