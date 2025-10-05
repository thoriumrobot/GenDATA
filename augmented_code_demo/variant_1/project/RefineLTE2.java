// Test case for issue #62:
// https://github.com/kelloggm/checker-framework/issues/62

    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class RefineLTE2 {

    @Positive
  protected int @MinLen(1) [] values;

    @Positive
  @LTEqLengthOf("values") int num_values;

    @Positive
  public void add(int elt) {
    @Positive
    if (num_values == values.length) {
    @Positive
      values = null;
      // :: error: (unary.increment)
    @Positive
      num_values++;
    @Positive
      return;
    @Positive
    }
    @Positive
    values[num_values] = elt;
    @Positive
    num_values++;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
