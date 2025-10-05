// Test case for https://github.com/typetools/checker-framework/issues/2452

    @Positive
import java.lang.reflect.Array;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
class Issue2452 {
    @Positive
  Object m1(Object[] a1) {
    @Positive
    if (Array.getLength(a1) > 0) {
    @Positive
      return Array.get(a1, 0);
    @Positive
    } else {
    @Positive
      return null;
    @Positive
    }
    @Positive
  }

    @Positive
  void m2() {
    @Positive
    int[] arr = {1, 2, 3};
    @Positive
    @LTEqLengthOf({"arr"}) int a = Array.getLength(arr);
    @Positive
  }

    @Positive
  void testMinLenSubtractPositive(String @MinLen(10) [] s) {
    @Positive
    @Positive int i1 = s.length - 9;
    @Positive
    @NonNegative int i0 = Array.getLength(s) - 10;
    // ::  error: (assignment)
    @Positive
    @NonNegative int im1 = Array.getLength(s) - 11;
    @Positive
  }

    @Positive
  void testLessThanLength(String[] s, @IndexOrHigh("#1") int i, @IndexOrHigh("#1") int j) {
    @Positive
    if (i < Array.getLength(s)) {
    @Positive
      @IndexFor("s") int in = i;
      // ::  error: (assignment)
    @Positive
      @IndexFor("s") int jn = j;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
