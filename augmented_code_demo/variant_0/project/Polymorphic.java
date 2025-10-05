    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyLowerBound;
    @Positive
import org.checkerframework.checker.index.qual.PolySameLen;
    @Positive
import org.checkerframework.checker.index.qual.PolyUpperBound;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.index.qual.SameLen;

    @Positive
public class Polymorphic {

  // Identity functions

    @Positive
  @PolyLowerBound int lbc_identity(@PolyLowerBound int a) {
    @Positive
    return a;
    @Positive
  }

    @Positive
  int @PolySameLen [] samelen_identity(int @PolySameLen [] a) {
    @Positive
    int @SameLen("a") [] x = a;
    @Positive
    return a;
    @Positive
  }

    @Positive
  @PolyUpperBound int ubc_identity(@PolyUpperBound int a) {
    @Positive
    return a;
    @Positive
  }

  // SameLen tests
    @Positive
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
    @Positive
    int[] banana;
    @Positive
    int @SameLen("a2") [] b = samelen_identity(a);
    // :: error: (assignment)
    @Positive
    int @SameLen("banana") [] c = samelen_identity(b);
    @Positive
  }

  // UpperBound tests
    @Positive
  void ubc_id(
    @Positive
      int[] a,
    @Positive
      int[] b,
    @Positive
      @LTLengthOf("#1") int ai,
    @Positive
      @LTEqLengthOf("#1") int al,
    @Positive
      @LTLengthOf({"#1", "#2"}) int abi,
    @Positive
      @LTEqLengthOf({"#1", "#2"}) int abl) {
    @Positive
    int[] c;

    @Positive
    @LTLengthOf("a") int ai1 = ubc_identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = ubc_identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = ubc_identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = ubc_identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = ubc_identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = ubc_identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = ubc_identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = ubc_identity(abl);
    @Positive
  }

  // LowerBound tests
    @Positive
  void lbc_id(@NonNegative int n, @Positive int p, @GTENegativeOne int g) {
    @Positive
    @NonNegative int an = lbc_identity(n);
    // :: error: (assignment)
    @Positive
    @Positive int bn = lbc_identity(n);

    @Positive
    @GTENegativeOne int ag = lbc_identity(g);
    // :: error: (assignment)
    @Positive
    @NonNegative int bg = lbc_identity(g);

    @Positive
    @Positive int ap = lbc_identity(p);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
