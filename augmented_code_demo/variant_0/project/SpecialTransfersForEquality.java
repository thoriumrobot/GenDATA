    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class SpecialTransfersForEquality {

    @Positive
  void gteN1Test(@GTENegativeOne int y) {
    @Positive
    int[] arr = new int[10];
    @Positive
    if (-1 != y) {
    @Positive
      @NonNegative int z = y;
    @Positive
      if (z < 10) {
    @Positive
        int k = arr[z];
    @Positive
      }
    @Positive
    }
    @Positive
  }

    @Positive
  void nnTest(@NonNegative int i) {
    @Positive
    if (i != 0) {
    @Positive
      @Positive int m = i;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
