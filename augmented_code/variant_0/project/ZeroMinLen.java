/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class ZeroMinLen {

    @Positive
  int @MinLen(1) [] nums;
    @Positive
  int[] nums2;

    @Positive
  @IndexFor("nums") int current_index;

    @Positive
  @IndexFor("nums2") int current_index2;

    @Positive
  void test() {
    @Positive
    current_index = 0;
    // :: error: (assignment)
    @Positive
    current_index2 = 0;
    @Positive
  }
    @Positive
}
