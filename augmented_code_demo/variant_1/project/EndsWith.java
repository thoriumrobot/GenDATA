// Test case for issue #56:
// https://github.com/kelloggm/checker-framework/issues/56

    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class EndsWith {

    @Positive
  void testEndsWith(String arg) {
    @Positive
    if (arg.endsWith("[]")) {
    @Positive
      @MinLen(2) String arg2 = arg;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
