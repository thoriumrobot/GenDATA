    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class NonNegArrayLength {

    @Positive
  public static void NonNegArrayLength(int @MinLen(4) [] arr) {
    @Positive
    @Positive int i = arr.length - 2;
    @Positive
  }
    @Positive
}
