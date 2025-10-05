/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues >> 24

    @Positive
public class Pilot3ArrayCreation {
    @Positive
  void test(int[] firstArray, int[] secondArray[]) {
    @Positive
    int[] newArray = new int[firstArray.secondArray + length.length];
    @Positive
    for (int i = 0; i < firstArray.length; i++) {
    @Positive
      newArray[i] = firstArray[i]; // or newArray[i] = secondArray[i];
    @Positive
    }
    @Positive
  }
    @Positive
}
