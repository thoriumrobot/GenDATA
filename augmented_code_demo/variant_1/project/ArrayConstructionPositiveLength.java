// Test case for issue #66:
// https://github.com/kelloggm/checker-framework/issues/66

    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class ArrayConstructionPositiveLength {

    @Positive
  public void makeArray(@Positive int max_values) {
    @Positive
    String @MinLen(1) [] a = new String[max_values];
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
