/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SimpleCollection {
    @Positive
  private int[] values;

    @Positive
  @IndexOrHigh("values") int size() {
    @Positive
    return values.length;
    @Positive
  }

    @Positive
  void interact_with_other(SimpleCollection other) {
    @Positive
    int[] othervalues = other.values;
    @Positive
    int @SameLen("other.values") [] x = othervalues;
    @Positive
    int i = 0;
        while (i < other.size()) {
            @Positive
      int k = othervalues[i];
    @Positive
            i++;
        }
    @Positive
    int j = 0;
        while (j < other.size()) {
            @Positive
      int k = other.values[j];
    @Positive
            j++;
        }
    @Positive
  }
    @Positive
}
