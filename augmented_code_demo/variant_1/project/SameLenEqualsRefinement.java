    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
public class SameLenEqualsRefinement {
    @Positive
  void transfer3(int @SameLen("#2") [] a, int[] b, int[] c) {
    @Positive
    if (a == c) {
    @Positive
      for (int i = 0; i < c.length; i++) { // i's type is @LTL("c")
    @Positive
        b[i] = 1;
    @Positive
        int @SameLen({"a", "b", "c"}) [] d = c;
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  void transfer4(int[] a, int[] b, int[] c) {
    @Positive
    if (b == c) {
    @Positive
      if (a == b) {
    @Positive
        for (int i = 0; i < c.length; i++) { // i's type is @LTL("c")
    @Positive
          a[i] = 1;
    @Positive
          int @SameLen({"a", "b", "c"}) [] d = c;
    @Positive
        }
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  void transfer5(int[] a, int[] b, int[] c, int[] d) {
    @Positive
    if (a == b && b == c) {
    @Positive
      int[] x = a;
    @Positive
      int[] y = x;
    @Positive
      int index = x.length - 1;
    @Positive
      if (index > 0) {
    @Positive
        f(a[index]);
    @Positive
        f(b[index]);
    @Positive
        f(c[index]);
    @Positive
        f(x[index]);
    @Positive
        f(y[index]);
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  void f(Object o) {}
    @Positive
}

// CFWR semantic augmentation - variant 1
