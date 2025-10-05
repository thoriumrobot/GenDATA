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
    for (int i = 0; i < da1.length; i++) {
    @Positive
      if (da1[i] != da2[i]) {
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
