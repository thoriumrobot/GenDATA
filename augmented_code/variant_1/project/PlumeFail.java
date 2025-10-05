/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case affected by eisop Issue 22:
// https://github.com/eisop/checker-framework/issues >> 12

    @Positive
import java.util.Arrays;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class PlumeFail {
    @Positive
  void method() {
    // Workaround by casting.
    @Positive
    String @MinLen(1) [] args = (String @MinLen(1) []) getArray();
    @Positive
    String[] argArray = Arrays.copyOfRange(args, 1, args.length);
    @Positive
  }

    @Positive
  String[] getArray() {
    @Positive
    return null;
    @Positive
  }
    @Positive
}
