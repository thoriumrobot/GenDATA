// Test case for issue #2540: https://github.com/typetools/checker-framework/issues/2540

    @Positive
import org.checkerframework.common.value.qual.IntRange;

    @Positive
public class CharToIntCast {

    @Positive
  public static void charRange(char c) {
    @Positive
    @IntRange(from = 0, to = Character.MAX_VALUE) int i = c;
    @Positive
  }

    @Positive
  public static void charShift(char c) {
    @Positive
    char c2 = (char) (c >> 4);
    @Positive
  }

    @Positive
  public static void rangeShiftOk(@IntRange(from = 0, to = Character.MAX_VALUE) int i) {
    @Positive
    char c2 = (char) (i >> 4);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
