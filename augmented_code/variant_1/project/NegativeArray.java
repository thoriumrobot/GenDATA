/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;

    @Positive
public class NegativeArray {

    @Positive
  public static void negativeArray(@GTENegativeOne int len) {
    // :: error: (array.length.negative)
    @Positive
    int[] arr = new int[len];
    @Positive
  }
    @Positive
}
