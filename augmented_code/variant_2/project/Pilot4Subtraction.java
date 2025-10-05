/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue 42: https://github.com/kelloggm/checker-framework/issues/42

// @skip-test until bug is fixed

    @Positive
public class Pilot4Subtraction {

    @Positive
  private static int[] getSecondHalf(int[] array) {
    @Positive
    int len = array.length / 2;
    @Positive
    int b = len - 1;
    @Positive
    int[] arr = new int[len];
    @Positive
    int a = 0;
        while (a < len) {
            @Positive
      arr[a] = array[b];
    @Positive
      b--;
    @Positive
            a++;
        }
    @Positive
    return arr;
    @Positive
  }
    @Positive
}
