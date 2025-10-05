    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class Index132 {
    @Positive
  public static String @ArrayLen({3, 4}) [] esc_quantify(String @ArrayLen({1, 2}) ... vars) {
    @Positive
    if (vars.length == 1) {
    @Positive
      return new String[] {"hello", vars[0], ")"};
    @Positive
    } else {
    @Positive
      return new String[] {"hello", vars[0], vars[1], ")"};
    @Positive
    }
    @Positive
  }
    @Positive
}
