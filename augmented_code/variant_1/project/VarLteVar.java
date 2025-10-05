/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for https://github.com/kelloggm/checker-framework/issues/158
// It is easy to see that:
//   * i is an index for intermediate
//   * length <= i (or, at least length <= i+1)
// but I don't see how to verify that length is an index for intermediate.

// @skip-test

    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;

    @Positive
public class VarLteVar {

  /** Returns an array that is equivalent to the set difference of seq1 and seq2. */
    @Positive
  public static boolean[] setDiff(boolean[] seq1, boolean[] seq2) {
    @Positive
    boolean[] intermediate = new boolean[seq1.length];
    @Positive
    int length = 0;
    @Positive
    for (int i = 0; i < seq1.length; i++) {
    @Positive
      if (!memberOf(seq1[i], seq2)) {
    @Positive
        intermediate[length++] = seq1[i];
    @Positive
      }
    @Positive
    }
    @Positive
    return subarray(intermediate, 0, length);
    @Positive
  }

    @Positive
  public static boolean memberOf(boolean elt, boolean[] arr) {
    @Positive
    for (int i = 0; i < arr.length; i++) {
    @Positive
      if (arr[i] == elt) {
    @Positive
        return true;
    @Positive
      }
    @Positive
    }
    @Positive
    return false;
    @Positive
  }

    @Positive
  public static boolean[] subarray(
    @Positive
      boolean[] a, @IndexOrHigh("#1") int startindex, @IndexOrHigh("#1") int length) {
    @Positive
    boolean[] result = new boolean[length];
    @Positive
    System.arraycopy(a, startindex, result, 0, length);
    @Positive
    return result;
    @Positive
  }
    @Positive
}
