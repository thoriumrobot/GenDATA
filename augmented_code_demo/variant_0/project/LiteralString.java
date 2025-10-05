// Test case for issue #55:
// https://github.com/kelloggm/checker-framework/issues/55

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class LiteralString {

    @Positive
  private static final String[] finalField = {"This", "is", "an", "array"};

    @Positive
  void testLiteralString() {
    @Positive
    @MinLen(10) String s = "This string is long enough";
    @Positive
  }

    @Positive
  void testLiteralArray() {
    @Positive
    String @MinLen(2) [] a = new String[] {"This", "array", "is", "long", "enough"};
    @Positive
    String @MinLen(2) [] b = finalField;
    @Positive
    @IndexFor("finalField") int i = 0;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
