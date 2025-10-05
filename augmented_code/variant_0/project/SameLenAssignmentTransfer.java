/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SameLenAssignmentTransfer {
    @Positive
  void transfer5(int @SameLen("#2") [] a, int[] b) {
    @Positive
    int[] c = a;
    @Positive
    for (int i = 0; i < c.length; i++) { // i's type is @LTL("c")
    @Positive
      b[i] = 1;
    @Positive
    }
    @Positive
  }
    @Positive
}
