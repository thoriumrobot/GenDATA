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
    for (int j = 0; j < a.length; j++) {
    @Positive
      a[j] = b[j];
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
