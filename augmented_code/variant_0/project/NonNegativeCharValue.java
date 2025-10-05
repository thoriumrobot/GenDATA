/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class NonNegativeCharValue {
    @Positive
  public static String toString(final @NonNegative Character ch) {
    @Positive
    return toString(ch.charValue());
    @Positive
  }
    @Positive
}
