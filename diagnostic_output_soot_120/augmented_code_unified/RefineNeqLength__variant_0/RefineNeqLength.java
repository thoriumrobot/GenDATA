// Test case for Issue panacekcz#12:
// https://github.com/panacekcz/checker-framework/issues/12

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTOMLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.common.value.qual.IntVal;

    @Positive
public class RefineNeqLength {
    @Positive
  void refineNeqLength(int[] array, @IndexOrHigh("#1") int i) {
    // Refines i <= array.length to i < array.length
    @Positive
    if (i != array.length) {
    @Positive
      refineNeqLengthMOne(array, i);
    @Positive
    }
    // No refinement
    @Positive
    if (i != array.length - 1) {
      // :: error: (argument)
    @Positive
      refineNeqLengthMOne(array, i);
    @Positive
    }
    @Positive
  }

    @Positive
  void refineNeqLengthMOne(int[] array, @IndexFor("#1") int i) {
    // Refines i < array.length to i < array.length - 1
    @Positive
    if (i != array.length - 1) {
    @Positive
      refineNeqLengthMTwo(array, i);
      // :: error: (argument)
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    @Positive
  }

    @Positive
  void refineNeqLengthMTwo(int[] array, @NonNegative @LTOMLengthOf("#1") int i) {
    // Refines i < array.length - 1 to i < array.length - 2
    @Positive
    if (i != array.length - 2) {
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    // No refinement
    @Positive
    if (i != array.length - 1) {
      // :: error: (argument)
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    @Positive
  }

    @Positive
  void refineNeqLengthMTwoNonLiteral(
    @Positive
      int[] array,
    @Positive
      @NonNegative @LTOMLengthOf("#1") int i,
    @Positive
      @IntVal(3) int c3,
    @Positive
      @IntVal({2, 3}) int c23) {
    // Refines i < array.length - 1 to i < array.length - 2
    @Positive
    if (i != array.length - (5 - c3)) {
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    // No refinement
    @Positive
    if (i != array.length - c23) {
      // :: error: (argument)
    @Positive
      refineNeqLengthMThree(array, i);
    @Positive
    }
    @Positive
  }

    @Positive
  @LTLengthOf(value = "#1", offset = "3") int refineNeqLengthMThree(
    @Positive
      int[] array, @NonNegative @LTLengthOf(value = "#1", offset = "2") int i) {
    // Refines i < array.length - 2 to i < array.length - 3
    @Positive
    if (i != array.length - 3) {
    @Positive
      return i;
    @Positive
    }
    // :: error: (return)
    @Positive
    return i;
    @Positive
  }

  // The same test for a string.
    @Positive
  @LTLengthOf(value = "#1", offset = "3") int refineNeqLengthMThree(
    @Positive
      String str, @NonNegative @LTLengthOf(value = "#1", offset = "2") int i) {
    // Refines i < str.length() - 2 to i < str.length() - 3
    @Positive
    if (i != str.length() - 3) {
    @Positive
      return i;
    @Positive
    }
    // :: error: (return)
    @Positive
    return i;
    @Positive
  }
    @Positive
}
