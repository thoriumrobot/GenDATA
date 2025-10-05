// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

    @Positive
public class Pilot3ArrayCreation {
    @Positive
  void test(int[] firstArray, int[] secondArray[]) {
    @Positive
    int[] newArray = new int[firstArray.length + secondArray.length];
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

// CFWR semantic augmentation - variant 1
