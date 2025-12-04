    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class RangeIndex {
    @Positive
  void foo(@IntRange(from = 0, to = 11) int x, int @MinLen(10) [] a) {
    // :: error: (array.access.unsafe.high.range)
    @Positive
    int y = a[x];
    @Positive
  }
    @Positive
}
