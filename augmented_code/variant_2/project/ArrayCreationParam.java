/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue 93: https://github.com/kelloggm/checker-framework/issues/93

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class ArrayCreationParam {

    @Positive
  public static void m1() {
    @Positive
    int n = 5;
    @Positive
    int[] a = new int[n + 1];
    // Index Checker correctly issues no warning on the lines below
    @Positive
    @LTLengthOf("a") int j = n;
    @Positive
    @IndexFor("a") int k = n;
    @Positive
    for (int i = 1; i <= n; i++) {
    @Positive
      int x = a[i];
    @Positive
    }
    @Positive
  }
    @Positive
}
