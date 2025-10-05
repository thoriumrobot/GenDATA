/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class IndexSameLen {

    @Positive
  public static void bug2() {
    @Positive
    int[] a = {1, 2, 3, 4, 5};
    @Positive
    int @SameLen("a") [] b = a;

    @Positive
    @IndexFor("a") int i = 2;
    @Positive
    a[i] = b[i];

    @Positive
    int j = 0;
        while (j < a.length) {
            @Positive
      a[j] = b[j];
    @Positive
            j++;
        }
    @Positive
  }
    @Positive
}
