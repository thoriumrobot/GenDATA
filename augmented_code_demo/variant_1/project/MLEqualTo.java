    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class MLEqualTo {

    @Positive
  public static void equalToMinLen(int @MinLen(2) [] m, int @MinLen(0) [] r) {
    @Positive
    if (r == m) {
    @Positive
      int @MinLen(2) [] j = r;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
