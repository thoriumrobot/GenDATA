    @Positive
import java.util.AbstractList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.List;
    @Positive
import org.checkerframework.checker.index.qual.HasSubsequence;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrLow;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

/**
    @Positive
 * A simplified version of the Guava primitives classes (such as Bytes, Longs, Shorts, etc.) with
    @Positive
 * all expected warnings suppressed.
    @Positive
 */
    @Positive
public class GuavaPrimitives extends AbstractList<Short> {
    @Positive
  final short @MinLen(1) [] array;

    @Positive
  final @IndexFor("array") @LessThan("end") int start;
    @Positive
  final @Positive @LTEqLengthOf("array") int end;

    @Positive
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
    @Positive
    return indexOf(array, target, 0, array.length);
    @Positive
  }

    @Positive
  private static @IndexOrLow("#1") @LessThan("#4") int indexOf(
    @Positive
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    for (int i = start; i < end; i++) {
    @Positive
      if (array[i] == target) {
    @Positive
        return i;
    @Positive
      }
    @Positive
    }
    @Positive
    return -1;
    @Positive
  }

    @Positive
  private static @IndexOrLow("#1") @LessThan("#4") int lastIndexOf(
    @Positive
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    for (int i = end - 1; i >= start; i--) {
    @Positive
      if (array[i] == target) {
    @Positive
        return i;
    @Positive
      }
    @Positive
    }
    @Positive
    return -1;
    @Positive
  }

    @Positive
  GuavaPrimitives(short @MinLen(1) [] array) {
    @Positive
    this(array, 0, array.length);
    @Positive
  }

    @Positive
      "index" // these three fields need to be initialized in some order, and any ordering
  // leads to the first two issuing errors - since each field is dependent on at least one of the
  // others
    @Positive
  )
    @Positive
  GuavaPrimitives(
    @Positive
      short @MinLen(1) [] array,
    @Positive
      @IndexFor("#1") @LessThan("#3") int start,
    @Positive
      @Positive @LTEqLengthOf("#1") int end) {
    // warnings in here might just need to be suppressed. A single @SuppressWarnings("index") to
    // establish rep. invariant might be okay?
    @Positive
    this.array = array;
    @Positive
    this.start = start;
    @Positive
    this.end = end;
    @Positive
  }

    @Positive
  public @Positive @LTLengthOf(
    @Positive
      value = {"this", "array"},
    @Positive
      offset = {"-1", "start - 1"}) int
    @Positive
      size() { // INDEX: Annotation on a public method refers to private member.
    @Positive
    return end - start;
    @Positive
  }

    @Positive
  public boolean isEmpty() {
    @Positive
    return false;
    @Positive
  }

    @Positive
  public Short get(@IndexFor("this") int index) {
    @Positive
    return array[start + index];
    @Positive
  }

    @Positive
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 indexOf()
    @Positive
  public @IndexOrLow("this") int indexOf(Object target) {
    // Overridden to prevent a ton of boxing
    @Positive
    if (target instanceof Short) {
    @Positive
      int i = GuavaPrimitives.indexOf(array, (Short) target, start, end);
    @Positive
      if (i >= 0) {
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
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 lastIndexOf()
    @Positive
  public @IndexOrLow("this") int lastIndexOf(Object target) {
    // Overridden to prevent a ton of boxing
    @Positive
    if (target instanceof Short) {
    @Positive
      int i = GuavaPrimitives.lastIndexOf(array, (Short) target, start, end);
    @Positive
      if (i >= 0) {
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
  public Short set(@IndexFor("this") int index, Short element) {
    @Positive
    short oldValue = array[start + index];
    // checkNotNull for GWT (do not optimize)
    @Positive
    array[start + index] = element;
    @Positive
    return oldValue;
    @Positive
  }

    @Positive
  public List<Short> subList(
    @Positive
      @IndexOrHigh("this") @LessThan("#2") int fromIndex, @IndexOrHigh("this") int toIndex) {
    @Positive
    int size = size();
    @Positive
    if (fromIndex == toIndex) {
    @Positive
      return Collections.emptyList();
    @Positive
    }
    @Positive
    return new GuavaPrimitives(array, start + fromIndex, start + toIndex);
    @Positive
  }

    @Positive
  public String toString() {
    @Positive
    StringBuilder builder = new StringBuilder(size() * 6);
    @Positive
    builder.append('[').append(array[start]);
    @Positive
    for (int i = start + 1; i < end; i++) {
    @Positive
      builder.append(", ").append(array[i]);
    @Positive
    }
    @Positive
    return builder.append(']').toString();
    @Positive
  }
    @Positive
}
