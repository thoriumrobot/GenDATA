// test case for issue 158: https://github.com/kelloggm/checker-framework/issues/158

// @skip-test until fixed

    @Positive
public class Pilot2HalfLength {
    @Positive
  private static int[] getFirstHalf(int[] array) {
    @Positive
    int[] firstHalf = new int[array.length / 2];
    @Positive
    for (int i = 0; i < firstHalf.length; i++) {
    @Positive
      firstHalf[i] = array[i];
    @Positive
    }
    @Positive
    return firstHalf;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
