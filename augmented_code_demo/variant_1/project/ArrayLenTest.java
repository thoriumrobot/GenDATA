    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class ArrayLenTest {
    @Positive
  public static String esc_quantify(String @ArrayLen({1, 2}) ... vars) {
    @Positive
    if (vars.length == 1) {
    @Positive
      return vars[0];
    @Positive
    } else {
    @Positive
      @IntVal({2}) int i = vars.length;
    @Positive
      String @ArrayLen({2}) [] a = vars;
    @Positive
      return vars[0] + vars[1];
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
