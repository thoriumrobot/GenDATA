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

// CFWR semantic augmentation - variant 0
