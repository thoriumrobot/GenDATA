/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue #68:
// https://github.com/kelloggm/checker-framework/issues/68

    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;

    @Positive
public class TestAgainstLength {

    @Positive
  protected int[] values;

  /** The number of active elements (equivalently, the first unused index). */
    @Positive
  @IndexOrHigh("values") int num_values;

    @Positive
  public void add(int elt) {
    @Positive
    if (num_values == values.length) {
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
