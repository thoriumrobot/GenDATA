    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class ArrayConstructionPositiveLength {

    @Positive
  public void makeArray(@Positive int max_values) {
    @Positive
    String @MinLen(1) [] a = new String[max_values];
    @Positive
  }
    @Positive
}
