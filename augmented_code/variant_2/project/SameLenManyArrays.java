/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
public class SameLenManyArrays {
    @Positive
  void transfer1(int @SameLen("#2") [] a, int[] b) {
    @Positive
    int[] c = new int[a.length];
    @Positive
    int i = 0;
        while (i < c.length) {
            // i's type is @LTL("c")
    @Positive
      b[i] = 1;
    @Positive
      int @SameLen({"a", "b", "c"}) [] d = c;
    @Positive
            i++;
        }
    @Positive
  }

    @Positive
  void transfer2(int @SameLen("#2") [] a, int[] b) {
    @Positive
    int i = 0;
        while (i < b.length) {
            // i's type is @LTL("b")
    @Positive
      a[i] = 1;
    @Positive
            i++;
        }
    @Positive
  }

    @Positive
  void transfer3(int @SameLen("#2") [] a, int[] b, int[] c) {
    @Positive
    if (a.length == c.length) {
    @Positive
      int i = 0;
        while (i < c.length) {
            // i's type is @LTL("c")
    @Positive
        b[i] = 1;
    @Positive
        int @SameLen({"a", "b", "c"}) [] d = c;
    @Positive
            i++;
        }
    @Positive
    }
    @Positive
  }

    @Positive
  void transfer4(int[] a, int[] b, int[] c) {
    @Positive
    if (b.length == c.length) {
    @Positive
      if (a.length == b.length) {
    @Positive
        int i = 0;
        while (i < c.length) {
            // i's type is @LTL("c")
    @Positive
          a[i] = 1;
    @Positive
          int @SameLen({"a", "b", "c"}) [] d = c;
    @Positive
            i++;
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
    if (a.length == b.length && b.length == c.length) {
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
