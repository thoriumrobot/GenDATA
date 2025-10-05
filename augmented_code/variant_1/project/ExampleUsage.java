/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    int i = 0;
        while (i < 5) {
            @Positive
      k = arr[i];
    @Positive
            i++;
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
    int i = -1;
        while (i < 4) {
            @Positive
      i++;
    @Positive
      k = arr[i];
    @Positive
            ;
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
    int i = -1;
        while (i < 5) {
            // :: error: (array.access.unsafe.low)
    @Positive
      k = arr[i];
    @Positive
            i++;
        }
    @Positive
  }
    @Positive
}
// a comment
