/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue #14:
// https://github.com/kelloggm/checker-framework/issues/14

    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class ArrayLength2 {
    @Positive
  public static void main(String[] args) {
    @Positive
    int N = 8;
    @Positive
    int @MinLen(8) [] Grid = new int[N];
    @Positive
    @LTLengthOf("Grid") int i = 0;
    @Positive
  }
    @Positive
}
