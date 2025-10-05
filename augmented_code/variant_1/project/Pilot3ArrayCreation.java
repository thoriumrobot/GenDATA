/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

    @Positive
public class Pilot3ArrayCreation {
    @Positive
  void test(int[] firstArray, int[] secondArray[]) {
    @Positive
    int[] newArray = new int[firstArray.length + secondArray.length];
    @Positive
    int i = 0;
        while (i < firstArray.length) {
            @Positive
      newArray[i] = firstArray[i]; // or newArray[i] = secondArray[i];
    @Positive
            i++;
        }
    @Positive
  }
    @Positive
}
