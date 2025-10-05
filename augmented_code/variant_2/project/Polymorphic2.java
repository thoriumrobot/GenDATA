/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
public class Polymorphic2 {
    @Positive
  public static boolean flag = false;

    @Positive
  int @PolySameLen [] mergeSameLen(int @PolySameLen [] a, int @PolySameLen [] b) {
    @Positive
    return flag ? a : b;
    @Positive
  }

    @Positive
  int[] array1 = new int[2];
    @Positive
  int[] array2 = new int[2];

    @Positive
  void testSameLen(int @SameLen("array1") [] a, int @SameLen("array2") [] b) {
    @Positive
    int[] x = mergeSameLen(a, b);
    // :: error: (assignment)
    @Positive
    int @SameLen("array1") [] y = mergeSameLen(a, b);
    @Positive
  }

    @Positive
  @PolyUpperBound int mergeUpperBound(@PolyUpperBound int a, @PolyUpperBound int b) {
    @Positive
    return flag ? a : b;
    @Positive
  }

  // UpperBound tests
    @Positive
  void testUpperBound(@LTLengthOf("array1") int a, @LTLengthOf("array2") int b) {
    @Positive
    int z = mergeUpperBound(a, b);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array1") int zz = mergeUpperBound(a, b);
    @Positive
  }

    @Positive
  void testUpperBound2(@LTLengthOf("array1") int a, @LTEqLengthOf("array1") int b) {
    @Positive
    @LTEqLengthOf("array1") int z = mergeUpperBound(a, b);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array1") int zz = mergeUpperBound(a, b);
    @Positive
  }

    @Positive
  @PolyLowerBound int mergeLowerBound(@PolyLowerBound int a, @PolyLowerBound int b) {
    @Positive
    return flag ? a : b;
    @Positive
  }

  // LowerBound tests
    @Positive
  void lbc_id(@NonNegative int n, @Positive int p) {
    @Positive
    @NonNegative int z = mergeLowerBound(n, p);
    // :: error: (assignment)
    @Positive
    @Positive int zz = mergeLowerBound(n, p);
    @Positive
  }
    @Positive
}
