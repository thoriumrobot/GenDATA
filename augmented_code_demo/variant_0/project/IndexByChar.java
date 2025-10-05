    @Positive
public class IndexByChar {
    @Positive
  public int m(char c) {
    @Positive
    int[] i = new int[128];
    @Positive
    if (c < 128) {
    @Positive
      return i[c];
    @Positive
    } else {
    @Positive
      return -1;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
