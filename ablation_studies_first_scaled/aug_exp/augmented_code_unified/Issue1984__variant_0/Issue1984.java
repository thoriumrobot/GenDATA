// Test case for Issue 1984
// https://github.com/typetools/checker-framework/issues/1984

    @Positive
import org.checkerframework.common.value.qual.IntRange;

    @Positive
public class Issue1984 {
    @Positive
  public int m(int[] a, @IntRange(from = 0, to = 12) int i) {
    // :: error: (array.access.unsafe.high.range)
    @Positive
    return a[i];
    @Positive
  }
    @Positive
}
