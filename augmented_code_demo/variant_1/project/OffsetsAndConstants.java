    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class OffsetsAndConstants {
    @Positive
  static int read(
    @Positive
      char[] a,
    @Positive
      @IndexOrHigh("#1") int off,
    @Positive
      @NonNegative @LTLengthOf(value = "#1", offset = "#2 - 1") int len) {
    @Positive
    int sum = 0;
    @Positive
    for (int i = 0; i < len; i++) {
    @Positive
      sum += a[i + off];
    @Positive
    }
    @Positive
    return sum;
    @Positive
  }

    @Positive
  public static void main(String[] args) {
    @Positive
    char[] a = new char[10];

    @Positive
    read(a, 5, 4);

    @Positive
    read(a, 5, 5);

    // :: error: (argument)
    @Positive
    read(a, 5, 6);

    // :: error: (argument)
    @Positive
    read(a, 5, 7);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
