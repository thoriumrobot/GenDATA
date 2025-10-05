    @Positive
import java.util.Arrays;
    @Positive
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class LTLengthOfPostcondition {

    @Positive
  Object[] array;

    @Positive
  @NonNegative @LTEqLengthOf("array") int end;

    @Positive
  public void shiftIndex(@NonNegative int x) {
    @Positive
    int newEnd = end - x;
    @Positive
    if (newEnd < 0) throw new RuntimeException();
    @Positive
    end = newEnd;
    @Positive
  }

    @Positive
  public void useShiftIndex(@NonNegative int x) {
    // :: error: (argument)
    @Positive
    Arrays.fill(array, end, end + x, null);
    @Positive
    shiftIndex(x);
    @Positive
    Arrays.fill(array, end, end + x, null);
    @Positive
  }

    @Positive
  public boolean tryShiftIndex(@NonNegative int x) {
    @Positive
    int newEnd = end - x;
    @Positive
    if (newEnd < 0) {
    @Positive
      return false;
    @Positive
    }
    @Positive
    end = newEnd;
    @Positive
    return true;
    @Positive
  }

    @Positive
  public void useTryShiftIndex(@NonNegative int x) {
    @Positive
    if (tryShiftIndex(x)) {
    @Positive
      Arrays.fill(array, end, end + x, null);
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
