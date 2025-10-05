/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// A test that caused a ClassCastException in the UpperBound Checker. Based on a
// function in MathPlume, discovered while minimizing another crash in WPI (hence
// why the function from MathPlume was changed to just return 0 in the first place...).

    @Positive
import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyUpperBound;

    @Positive
public final class MathPlumeClasscastCrash {

    @Positive
  public static @NonNegative @LessThan("#2") @PolyUpperBound long modPositive(
    @Positive
      long x, @PolyUpperBound long y) {
    @Positive
    return 0;
    @Positive
  }
    @Positive
}
