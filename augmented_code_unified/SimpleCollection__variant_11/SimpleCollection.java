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
    for (int i = 0; i < other.size(); i++) {
    @Positive
      int k = othervalues[i];
    @Positive
    }
    @Positive
    for (int j = 0; j < other.size(); j++) {
    @Positive
      int k = other.values[j];
    @Positive
    }
    @Positive
  }
    @Positive
}
