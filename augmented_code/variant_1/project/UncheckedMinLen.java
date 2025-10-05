/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

// test case for kelloggm#183: https://github.com/kelloggm/checker-framework/issues/183

    @Positive
public class UncheckedMinLen {
    @Positive
  void addToNonNegative(@NonNegative int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[1 + l];
    @Positive
    o[99] = v;
    @Positive
  }

    @Positive
  void addToPositive(@Positive int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[1 + l];
    @Positive
    o[99] = v;
    @Positive
  }

    @Positive
  void addToUnboundedIntRange(@IntRange(from = 0) int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[1 + l];
    @Positive
    o[99] = v;
    @Positive
  }

  // Similar code that correctly gives warnings
    @Positive
  void addToPositiveOK(@NonNegative int l, Object v) {
    @Positive
    Object[] o = new Object[1 + l];
    // :: error: (array.access.unsafe.high.constant)
    @Positive
    o[99] = v;
    @Positive
  }

    @Positive
  void addToBoundedIntRangeOK(@IntRange(from = 0, to = 1) int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[1 + l];
    @Positive
    o[99] = v;
    @Positive
  }

    @Positive
  void subtractFromPositiveOK(@Positive int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[l - 1];
    @Positive
    o[99] = v;
    @Positive
  }
    @Positive
}
