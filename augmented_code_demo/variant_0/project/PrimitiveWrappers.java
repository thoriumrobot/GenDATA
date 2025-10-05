// Test for issue 65: https://github.com/kelloggm/checker-framework/issues/65

    @Positive
import org.checkerframework.checker.index.qual.*;

// This test ensures that the checker functions on primitive wrappers in
// addition to literal primitives. Primarily it focuses on Integer/int.

    @Positive
public class PrimitiveWrappers {

    @Positive
  void int_Integer_access_equivalent(@IndexFor("#3") Integer i, @IndexFor("#3") int j, int[] a) {
    @Positive
    a[i] = a[j];
    @Positive
  }

    @Positive
  void array_creation(@NonNegative Integer i, @NonNegative int j) {
    @Positive
    int[] a = new int[j];
    @Positive
    int[] b = new int[i];
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
