/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for https://github.com/kelloggm/checker-framework/issues/154
// This class wraps an array, but doesn't expose the array in its public interface. This test
// ensures that indexes for this new collection can be annotated as if the collection were an array.

// Note that there is a copy of this code in the manual in index-checker.tex. If this code is
// updated, you MUST update that copy, as well.

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.SameLen;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

/** ArrayWrapper is a fixed-size generic collection. */
    @Positive
public class ArrayWrapper<T> {
    @Positive
  private final Object @SameLen("this") [] delegate;

    @Positive
  ArrayWrapper(@NonNegative int size) {
    @Positive
    delegate = new Object[size];
    @Positive
  }

    @Positive
  public @LengthOf("this") int size() {
    @Positive
    return delegate.length;
    @Positive
  }

    @Positive
  public void set(@IndexFor("this") int index, T obj) {
    @Positive
    delegate[index] = obj;
    @Positive
  }

    @Positive
  public T get(@IndexFor("this") int index) {
    @Positive
    return (T) delegate[index];
    @Positive
  }

    @Positive
  public static void clearIndex1(ArrayWrapper<? extends Object> a, @IndexFor("#1") int i) {
    @Positive
    a.set(i, null);
    @Positive
  }

    @Positive
  public static void clearIndex2(ArrayWrapper<? extends Object> a, int i) {
    @Positive
    if (0 <= i && i < a.size()) {
    @Positive
      a.set(i, null);
    @Positive
    }
    @Positive
  }

    @Positive
  public static void clearIndex3(ArrayWrapper<? extends Object> a, @NonNegative int i) {
    @Positive
    if (i < a.size()) {
    @Positive
      a.set(i, null);
    @Positive
    }
    @Positive
  }

  // The following methods are tests that sequence annotations work correctly with
  // user-defined sequence types.

    @Positive
  public static Object testSameLen(
    @Positive
      @IndexFor("#1") int i) {
    @Positive
    return b.get(i);
    @Positive
  }

    @Positive
  public static Object testMinLen(@MinLen(3) ArrayWrapper<? extends Object> a) {
    @Positive
    return a.get(2);
    @Positive
  }
    @Positive
}
