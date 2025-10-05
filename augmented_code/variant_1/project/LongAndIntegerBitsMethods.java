/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class LongAndIntegerBitsMethods {
    @Positive
  void caseInteger(
    @Positive
      int index, int @MinLen(33) [] arr1, int @MinLen(33) [] arr2, int val1, int val2) {
    @Positive
    arr1[Integer.numberOfLeadingZeros(index)] = val1;
    @Positive
    arr2[Integer.numberOfTrailingZeros(index)] = val2;
    @Positive
  }

    @Positive
  void caseLong(int index, int @MinLen(65) [] arr1, int @MinLen(65) [] arr2, int val1, int val2) {
    @Positive
    arr1[Long.numberOfLeadingZeros(index)] = val1;
    @Positive
    arr2[Long.numberOfTrailingZeros(index)] = val2;
    @Positive
  }
    @Positive
}
