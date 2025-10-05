    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class Issue2505 {
    @Positive
  public static void warningIfStatement(int @MinLen(1) [] a) {
    @Positive
    int i = a.length;
    @Positive
    if (--i >= 0) {
    @Positive
      a[i] = 0;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
