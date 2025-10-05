/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class Issue2029 {
    @Positive
  void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
    @Positive
    char[] arr = new char[size];
    @Positive
    arr[index] = val;
    @Positive
  }

    @Positive
  void LessThanOffsetLowerBound(
    @Positive
      int[] array, @NonNegative @LTLengthOf("#1") int n, @NonNegative @LessThan("#2 + 1") int k) {
    @Positive
    array[n - k] = 10;
    @Positive
  }

    @Positive
  void LessThanOffsetUpperBound(
    @Positive
      @NonNegative int n,
    @Positive
      @NonNegative @LessThan("#1 + 1") int k,
    @Positive
      @NonNegative int size,
    @Positive
      @NonNegative @LessThan("#3 + 1") int index) {
    @Positive
    @NonNegative int m = n - k;
    @Positive
    int[] arr = new int[size];
    // :: error: (unary.increment)
    @Positive
    for (; index < arr.length - 1; index++) {
    @Positive
      arr[index] = 10;
    @Positive
    }
    @Positive
  }
    @Positive
}
