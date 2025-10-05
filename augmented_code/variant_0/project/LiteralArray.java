/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// test case for issue #67:
// https://github.com/kelloggm/checker-framework/issues/67

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class LiteralArray {

    @Positive
  private static final String[] timeFormat = {
    @Positive
    ("#.#"), ("#.#"), ("#.#"), ("#.#"), ("#.#"),
    @Positive
  };

    @Positive
  public String format() {
    @Positive
    return format(1);
    @Positive
  }

    @Positive
  public String format(@IndexFor("LiteralArray.timeFormat") int digits) {
    @Positive
    return "";
    @Positive
  }
    @Positive
}
