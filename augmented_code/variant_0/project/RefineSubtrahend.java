/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for kelloggm 215
// https://github.com/kelloggm/checker-framework/issues/215

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class RefineSubtrahend {
    @Positive
  void withConstant(int[] a, @NonNegative int l) {
    @Positive
    if (a.length - l > 10) {
    @Positive
      int x = a[l + 10];
    @Positive
    }
    @Positive
    if (a.length - 10 > l) {
    @Positive
      int x = a[l + 10];
    @Positive
    }
    @Positive
    if (a.length - l >= 10) {
      // :: error: (array.access.unsafe.high)
    @Positive
      int x = a[l + 10];
    @Positive
      int x1 = a[l + 9];
    @Positive
    }
    @Positive
  }

    @Positive
  void withVariable(int[] a, @NonNegative int l, @NonNegative int j, @NonNegative int k) {
    @Positive
    if (a.length - l > j) {
    @Positive
      if (k <= j) {
    @Positive
        int x = a[l + k];
    @Positive
      }
    @Positive
    }
    @Positive
    if (a.length - j > l) {
    @Positive
      if (k <= j) {
    @Positive
        int x = a[l + k];
    @Positive
      }
    @Positive
    }
    @Positive
    if (a.length - j >= l) {
    @Positive
      if (k <= j) {
        // :: error: (array.access.unsafe.high)
    @Positive
        int x = a[l + k];
        // :: error: (array.access.unsafe.low)
    @Positive
        int x1 = a[l + k - 1];
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  void cases(int[] a, @NonNegative int l) {
    @Positive
    if (a.length - l == 1) {
            @Positive
int x = a[l];
@Positive
@Positive
        } else if (a.length - l == 2) {
            @Positive
int y = a[l + 1];
@Positive
@Positive
        }
    @Positive
  }
    @Positive
}
