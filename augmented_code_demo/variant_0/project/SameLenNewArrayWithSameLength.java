    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SameLenNewArrayWithSameLength {
    @Positive
  public void m1(int[] a) {
    @Positive
    int @SameLen("a") [] b = new int[a.length];
    @Positive
  }

    @Positive
  public void m2(int[] a, int @SameLen("#1") [] b) {
    @Positive
    int @SameLen({"a", "b"}) [] c = new int[b.length];
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
