    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class ErrorMessageCheck {
    @Positive
  @NonNegative int size;
    @Positive
  int[] vDown = new int[size];

    @Positive
  void method3(@NonNegative int size, @NonNegative int value) {
    @Positive
    this.size = size;
    @Positive
    this.vDown = new int[this.size];
    // :: error: (array.access.unsafe.high)
    @Positive
    vDown[1 + value] = 10;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
