/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// test case for issue 163: https://github.com/kelloggm/checker-framework/issues/163

    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyUpperBound;

    @Positive
public class UBPoly {
    @Positive
  public static void main(String[] args) {
    @Positive
    char[] a = new char[10];
    @Positive
    poly(a, 100);
    @Positive
  }

    @Positive
  public static void poly(char[] a, @NonNegative @PolyUpperBound int i) {
    // :: error: (argument)
    @Positive
    access(a, i);
    @Positive
  }

    @Positive
  public static void access(char[] a, @NonNegative @LTLengthOf("#1") int j) {
    @Positive
    char c = a[j];
    @Positive
  }
    @Positive
}
