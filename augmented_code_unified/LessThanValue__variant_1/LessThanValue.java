    @Positive
package lessthan;

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.common.value.qual.*;

// Test for LessThanChecker
    @Positive
public class LessThanValue {

    @Positive
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
    @Positive
    @LessThan("x") int q = a;
    // :: error: (assignment)
    @Positive
    int r = b;
    @Positive
  }

    @Positive
  public static boolean flag;

    @Positive
  void lub(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
    @Positive
    @LessThan("x") int r = flag ? a : b;
    // :: error: (assignment)
    @Positive
    int s = flag ? a : b;
    @Positive
  }

    @Positive
  void transitive(int a, int b, int c) {
    @Positive
    if (a < b) {
    @Positive
      if (b < c) {
        // :: error: (assignment)
    @Positive
        @LessThan("c") int x = a;
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  void calls() {
    @Positive
    isLessThan(0, 1);
    @Positive
    isLessThanOrEqual(0, 0);
    @Positive
  }

    @Positive
  void isLessThan(@LessThan("#2") @NonNegative int start, int end) {
    @Positive
    @NonNegative int x = end - start - 1;
    @Positive
    @Positive int y = end - start;
    @Positive
  }

    @Positive
  @NonNegative int isLessThanOrEqual(@LessThan("#2 + 1") @NonNegative int start, int end) {
    @Positive
    return end - start;
    @Positive
  }

    @Positive
  public void setMaximumItemCount(int maximum) {
    @Positive
    if (maximum < 0) {
    @Positive
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    @Positive
    }
    @Positive
    int count = getCount();
    @Positive
    if (count > maximum) {
    @Positive
      @Positive int y = count - maximum;
    @Positive
      @NonNegative int deleteIndex = count - maximum - 1;
    @Positive
    }
    @Positive
  }

    @Positive
  int getCount() {
    @Positive
    throw new RuntimeException();
    @Positive
  }

    @Positive
  void method(@NonNegative int m) {
    @Positive
    boolean[] has_modulus = new boolean[m];
    @Positive
    @LessThan("m") int x = foo(m);
    @Positive
    @IndexFor("has_modulus") int rem = foo(m);
    @Positive
  }

    @Positive
  @LessThan("#1") @NonNegative int foo(int in) {
    @Positive
    throw new RuntimeException();
    @Positive
  }

    @Positive
  void test(int maximum, int count) {
    @Positive
    if (maximum < 0) {
    @Positive
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    @Positive
    }
    @Positive
    if (count > maximum) {
    @Positive
      int deleteIndex = count - maximum - 1;
      // TODO: shouldn't error
      // :: error: (argument)
    @Positive
      isLessThanOrEqual(0, deleteIndex);
    @Positive
    }
    @Positive
  }

    @Positive
  void count(int count) {
    @Positive
    if (count > 0) {
    @Positive
      if (count % 2 == 1) {

    @Positive
      } else {
        // TODO: improve value checker
        // :: error: (assignment)
    @Positive
        @IntRange(from = 0) int countDivMinus = count / 2 - 1;
        // Reasign to update the value in the store.
    @Positive
        countDivMinus = countDivMinus;
        // :: error: (argument)
    @Positive
        isLessThan(0, countDivMinus);
    @Positive
        isLessThanOrEqual(0, countDivMinus);
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  static @NonNegative @LessThan("#2 + 1") int expandedCapacity(
    @Positive
      @NonNegative int oldCapacity, @NonNegative int minCapacity) {
    @Positive
    if (minCapacity < 0) {
    @Positive
      throw new AssertionError("cannot store more than MAX_VALUE elements");
    @Positive
    }
    // careful of overflow!
    @Positive
    int newCapacity = oldCapacity + (oldCapacity >> 1) + 1; // expand by %50
    @Positive
    if (newCapacity < minCapacity) {
    @Positive
      newCapacity = Integer.highestOneBit(minCapacity - 1) << 1;
    @Positive
    }
    @Positive
    if (newCapacity < 0) {
    @Positive
      newCapacity = Integer.MAX_VALUE;
      // guaranteed to be >= newCapacity
    @Positive
    }
    // :: error: (return)
    @Positive
    return newCapacity;
    @Positive
  }
    @Positive
}
