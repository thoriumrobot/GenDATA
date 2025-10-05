/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for eisop Issue 22:
// https://github.com/eisop/checker-framework/issues/22

    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
abstract class PlumeFailMin {
    @Positive
  void ok() {
    @Positive
    String @MinLen(1) [] args = getArrayOk();
    @Positive
    @IndexOrHigh("args") int x = 1;
    @Positive
  }

    @Positive
  abstract String @MinLen(1) [] getArrayOk();

    @Positive
  void fail() {
    // Workaround by casting.
    @Positive
    String @MinLen(1) [] args = (String @MinLen(1) []) getArrayFail();
    @Positive
    @IndexOrHigh("args") int x = 1;
    @Positive
  }

    @Positive
  abstract String[] getArrayFail();
    @Positive
}
