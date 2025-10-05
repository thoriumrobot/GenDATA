// Test case for Issue 167:
// https://github.com/kelloggm/checker-framework/issues/167

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LTOMLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class Index167 {
    @Positive
  static void fn1(int[] arr, @IndexFor("#1") int i) {
    @Positive
    if (i >= 33) {
      // :: error: (argument)
    @Positive
      fn2(arr, i);
    @Positive
    }
    @Positive
    if (i > 33) {
      // :: error: (argument)
    @Positive
      fn2(arr, i);
    @Positive
    }
    @Positive
    if (i != 33) {
      // :: error: (argument)
    @Positive
      fn2(arr, i);
    @Positive
    }
    @Positive
  }

    @Positive
  static void fn2(int[] arr, @NonNegative @LTOMLengthOf("#1") int i) {
    @Positive
    int c = arr[i + 1];
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
