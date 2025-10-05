    @Positive
package lessthan;

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrLow;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.dataflow.qual.Pure;

// This class has a similar implementation to several Immutable*Array class in Guava,
// such as com.google.common.primitives.ImmutableDoubleArray.
    @Positive
public class LessThanCustomCollection {
  // This object is a subset of array. So, if something is an index for "this"
  // then it is >= start and < end.
    @Positive
  private final int[] array;
    @Positive
  private final @IndexOrHigh("array") @LessThan("end + 1") int start;
    @Positive
  private final @LTLengthOf(
    @Positive
      value = {"array", "this"},
    @Positive
      offset = {" - 1", "- start"}) int end;

    @Positive
  private LessThanCustomCollection(int[] array) {
    @Positive
    this(array, 0, array.length);
    @Positive
  }

    @Positive
  private LessThanCustomCollection(
    @Positive
      int[] array, @IndexOrHigh("#1") @LessThan("#3 + 1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    this.array = array;
    // can't est. that end - start is the length of this.
    // :: error: (assignment)
    @Positive
    this.end = end;
    // start is @LessThan(end + 1) but should be @LessThan(this.end + 1)
    // :: error: (assignment)
    @Positive
    this.start = start;
    @Positive
  }

    @Positive
  public @LengthOf("this") int length() {
    @Positive
    return end - start;
    @Positive
  }

    @Positive
  public double get(@IndexFor("this") int index) {
    // TODO: This is a bug.
    // :: error: (argument)
    @Positive
    checkElementIndex(index, length());
    // Because index is an index for "this" the index + start
    // must be an index for array.
    // :: error: (array.access.unsafe.high)
    @Positive
    return array[start + index];
    @Positive
  }

    @Positive
  public static @NonNegative int checkElementIndex(
    @Positive
      @LessThan("#2") @NonNegative int index, @NonNegative int size) {
    @Positive
    if (index < 0 || index >= size) {
    @Positive
      throw new IndexOutOfBoundsException();
    @Positive
    }
    @Positive
    return index;
    @Positive
  }

    @Positive
  public @IndexOrLow("this") int indexOf(double target) {
    @Positive
    for (int i = start; i < end; i++) {
    @Positive
      if (areEqual(array[i], target)) {
        // Don't know that it is greater than start.
        // :: error: (return)
    @Positive
        return i - start;
    @Positive
      }
    @Positive
    }
    @Positive
    return -1;
    @Positive
  }

    @Positive
  static boolean areEqual(int item, double target) {
    // implementation not relevant
    @Positive
    return true;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
