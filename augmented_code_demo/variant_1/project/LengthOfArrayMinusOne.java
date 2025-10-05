    @Positive
public class LengthOfArrayMinusOne {
    @Positive
  void test(int[] arr) {
    // :: error: (array.access.unsafe.low)
    @Positive
    int i = arr[arr.length - 1];

    @Positive
    if (arr.length > 0) {
    @Positive
      int j = arr[arr.length - 1];
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
