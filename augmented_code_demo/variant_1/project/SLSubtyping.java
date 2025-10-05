    @Positive
import org.checkerframework.checker.index.qual.SameLen;

// This test checks whether the SameLen type system works as expected.

    @Positive
public class SLSubtyping {
    @Positive
  int[] f = {1};

    @Positive
  void subtype(int @SameLen("#2") [] a, int[] b) {
    @Positive
    int @SameLen({"a", "b"}) [] c = a;

    // :: error: (assignment)
    @Positive
    int @SameLen("c") [] q = {1, 2};
    @Positive
    int @SameLen("c") [] d = q;

    // :: error: (assignment)
    @Positive
    int @SameLen("f") [] e = a;
    @Positive
  }

    @Positive
  void subtype2(int[] a, int @SameLen("#1") [] b) {
    @Positive
    a = b;
    @Positive
    int @SameLen("b") [] c = b;
    @Positive
    int @SameLen("f") [] d = f;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
