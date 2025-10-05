/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue 98: https://github.com/kelloggm/checker-framework/issues/98

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SubtractingNonNegatives {
    @Positive
  public static void m4(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    int k = i;
    @Positive
    if (k >= j) {
    @Positive
      @IndexFor("a") int y = k;
    @Positive
    }
    @Positive
    for (k = i; k >= j; k -= j) {
    @Positive
      @IndexFor("a") int x = k;
    @Positive
    }
    @Positive
  }

    @Positive
  void test(int[] a, @Positive int y) {
    @Positive
    @LTLengthOf("a") int x = a.length - 1;
    @Positive
        value = {"a", "a"},
    @Positive
        offset = {"0", "y"})
    @Positive
    int z = x - y;
    @Positive
    a[z + y] = 0;
    @Positive
  }

    @Positive
  void test2(int[] a, @Positive int y) {
    @Positive
    @LTLengthOf("a") int x = a.length - 1;
    @Positive
    int z = x - y;
    @Positive
    a[z + y] = 0;
    @Positive
  }
    @Positive
}
