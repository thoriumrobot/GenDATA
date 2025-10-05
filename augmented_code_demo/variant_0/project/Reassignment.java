// @skip-test until we solve the underlying issue. The code that caused this to pass has been
// removed, because an incomplete solution that masks the problem but still permits some unsoundness
// is worse than no solution and an obvious issue.

    @Positive
public class Reassignment {
    @Positive
  void test(int[] arr, int i) {
    @Positive
    if (i > 0 && i < arr.length) {
    @Positive
      arr = new int[0];
      // :: error: (array.access.unsafe.high)
    @Positive
      int j = arr[i];
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
