    @Positive
public class ExampleUsage {
  /**
    @Positive
   * this class contains a set of test methods that are supposed to show how the lowerbound checker
    @Positive
   * should work in practice. They contain no annotations - the only test is whether or not it
    @Positive
   * alarms on particular code constructs that are or are not safe
    @Positive
   */
    @Positive
  void safe_loop_const() {
    @Positive
    int[] arr = new int[5];
    @Positive
    int k;
    @Positive
    for (int i = 0; i < 5; i++) {
    @Positive
      k = arr[i];
    @Positive
    }
    @Positive
  }

    @Positive
  void safe_loop_spooky() {
    @Positive
    int[] arr = new int[5];
    @Positive
    int k;
    @Positive
    for (int i = -1; i < 4; ) {
    @Positive
      i++;
    @Positive
      k = arr[i];
    @Positive
    }
    @Positive
  }

    @Positive
  void obviously_unsafe_loop() {
    @Positive
    int[] arr = new int[5];
    @Positive
    int k;
    @Positive
    for (int i = -1; i < 5; i++) {
      // :: error: (array.access.unsafe.low)
    @Positive
      k = arr[i];
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

// CFWR semantic augmentation - variant 1
