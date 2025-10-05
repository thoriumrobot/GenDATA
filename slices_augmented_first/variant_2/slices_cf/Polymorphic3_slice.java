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
    @LTLengthOf("a") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class Polymorphic3 {

  // Identity functions

    @Positive
  @PolyIndex int identity(@PolyIndex int a) {
    @Positive
    return a;
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
    @LTLengthOf("a") int identity(ai) = identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

  // LowerBound tests
    @Positive
  void lbc_id(@NonNegative int n, @Positive int p, @GTENegativeOne int g) {
    @Positive
    @NonNegative int an = identity(n);
    // :: error: (assignment)
    @Positive
    @Positive int bn = identity(n);

    @Positive
    @GTENegativeOne int ag = identity(g);
    // :: error: (assignment)
    @Positive
    @NonNegative int bg = identity(g);

    @Positive
    @Positive int ap = identity(p);
    @Positive
  }
    @Positive
}

    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class Polymorphic3 {

  // Identity functions

    @Positive
  @PolyIndex int identity(@PolyIndex int a) {
    @Positive
    return a;
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
    @LTLengthOf("a") int ai1 = identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int identity(ai) = identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

  // LowerBound tests
    @Positive
  void lbc_id(@NonNegative int n, @Positive int p, @GTENegativeOne int g) {
    @Positive
    @NonNegative int an = identity(n);
    // :: error: (assignment)
    @Positive
    @Positive int bn = identity(n);

    @Positive
    @GTENegativeOne int ag = identity(g);
    // :: error: (assignment)
    @Positive
    @NonNegative int bg = identity(g);

    @Positive
    @Positive int ap = identity(p);
    @Positive
  }
    @Positive
}


    @Positive
    @LTEqLengthOf("a") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class Polymorphic3 {

  // Identity functions

    @Positive
  @PolyIndex int identity(@PolyIndex int a) {
    @Positive
    return a;
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
    @LTLengthOf("a") int ai1 = identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = identity(ai);

    @Positive
    @LTEqLengthOf("a") int identity(al) = identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

  // LowerBound tests
    @Positive
  void lbc_id(@NonNegative int n, @Positive int p, @GTENegativeOne int g) {
    @Positive
    @NonNegative int an = identity(n);
    // :: error: (assignment)
    @Positive
    @Positive int bn = identity(n);

    @Positive
    @GTENegativeOne int ag = identity(g);
    // :: error: (assignment)
    @Positive
    @NonNegative int bg = identity(g);

    @Positive
    @Positive int ap = identity(p);
    @Positive
  }
    @Positive
}

    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class Polymorphic3 {

  // Identity functions

    @Positive
  @PolyIndex int identity(@PolyIndex int a) {
    @Positive
    return a;
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
    @LTLengthOf("a") int ai1 = identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int identity(al) = identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

  // LowerBound tests
    @Positive
  void lbc_id(@NonNegative int n, @Positive int p, @GTENegativeOne int g) {
    @Positive
    @NonNegative int an = identity(n);
    // :: error: (assignment)
    @Positive
    @Positive int bn = identity(n);

    @Positive
    @GTENegativeOne int ag = identity(g);
    // :: error: (assignment)
    @Positive
    @NonNegative int bg = identity(g);

    @Positive
    @Positive int ap = identity(p);
    @Positive
  }
    @Positive
}


    @Positive
    @LTLengthOf({"a", "b"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class Polymorphic3 {

  // Identity functions

    @Positive
  @PolyIndex int identity(@PolyIndex int a) {
    @Positive
    return a;
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
    @LTLengthOf("a") int ai1 = identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int identity(abi) = identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

  // LowerBound tests
    @Positive
  void lbc_id(@NonNegative int n, @Positive int p, @GTENegativeOne int g) {
    @Positive
    @NonNegative int an = identity(n);
    // :: error: (assignment)
    @Positive
    @Positive int bn = identity(n);

    @Positive
    @GTENegativeOne int ag = identity(g);
    // :: error: (assignment)
    @Positive
    @NonNegative int bg = identity(g);

    @Positive
    @Positive int ap = identity(p);
    @Positive
  }
    @Positive
}

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class Polymorphic3 {

  // Identity functions

    @Positive
  @PolyIndex int identity(@PolyIndex int a) {
    @Positive
    return a;
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
    @LTLengthOf("a") int ai1 = identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int identity(abi) = identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

  // LowerBound tests
    @Positive
  void lbc_id(@NonNegative int n, @Positive int p, @GTENegativeOne int g) {
    @Positive
    @NonNegative int an = identity(n);
    // :: error: (assignment)
    @Positive
    @Positive int bn = identity(n);

    @Positive
    @GTENegativeOne int ag = identity(g);
    // :: error: (assignment)
    @Positive
    @NonNegative int bg = identity(g);

    @Positive
    @Positive int ap = identity(p);
    @Positive
  }
    @Positive
}


    @Positive
    @LTEqLengthOf({"a", "b"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class Polymorphic3 {

  // Identity functions

    @Positive
  @PolyIndex int identity(@PolyIndex int a) {
    @Positive
    return a;
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
    @LTLengthOf("a") int ai1 = identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int identity(abl) = identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

  // LowerBound tests
    @Positive
  void lbc_id(@NonNegative int n, @Positive int p, @GTENegativeOne int g) {
    @Positive
    @NonNegative int an = identity(n);
    // :: error: (assignment)
    @Positive
    @Positive int bn = identity(n);

    @Positive
    @GTENegativeOne int ag = identity(g);
    // :: error: (assignment)
    @Positive
    @NonNegative int bg = identity(g);

    @Positive
    @Positive int ap = identity(p);
    @Positive
  }
    @Positive
}

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class Polymorphic3 {

  // Identity functions

    @Positive
  @PolyIndex int identity(@PolyIndex int a) {
    @Positive
    return a;
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
    @LTLengthOf("a") int ai1 = identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int identity(abl) = identity(abl);
    @Positive
  }

  // LowerBound tests
    @Positive
  void lbc_id(@NonNegative int n, @Positive int p, @GTENegativeOne int g) {
    @Positive
    @NonNegative int an = identity(n);
    // :: error: (assignment)
    @Positive
    @Positive int bn = identity(n);

    @Positive
    @GTENegativeOne int ag = identity(g);
    // :: error: (assignment)
    @Positive
    @NonNegative int bg = identity(g);

    @Positive
    @Positive int ap = identity(p);
    @Positive
  }
    @Positive
}

    @Positive
  }
