/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import java.util.Arrays;
    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
public class DaikonCrash {
    @Positive
  void method(Object[] a1) {
    @Positive
    int[] u = union(new int[] {}, new int[] {});
    @Positive
    Arrays.sort(u);
    @Positive
  }

    @Positive
  private int[] union(int[] ints, int[] ints1) {
    @Positive
    throw new RuntimeException();
    @Positive
  }
    @Positive
}
