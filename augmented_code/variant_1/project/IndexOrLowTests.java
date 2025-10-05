/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrLow;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class IndexOrLowTests {
    @Positive
  int[] array = {1, 2};

    @Positive
  @IndexOrLow("array") int index = -1;

    @Positive
  void test() {

    @Positive
    if (index != -1) {
    @Positive
      array[index] = 1;
    @Positive
    }

    @Positive
    @IndexOrHigh("array") int y = index + 1;
    // :: error: (array.access.unsafe.high)
    @Positive
    array[y] = 1;
    @Positive
    if (y < array.length) {
    @Positive
      array[y] = 1;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    index = array.length;
    @Positive
  }

    @Positive
  void test2(@LTLengthOf("array") @GTENegativeOne int param) {
    @Positive
    index = array.length - 1;
    @Positive
    @LTLengthOf("array") @GTENegativeOne int x = index;
    @Positive
    index = param;
    @Positive
  }
    @Positive
}
