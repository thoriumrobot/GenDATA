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
    for (int a = 0; a < len; a++) {
    @Positive
      arr[a] = array[b];
    @Positive
      b--;
    @Positive
    }
    @Positive
    return arr;
    @Positive
  }
    @Positive
}
