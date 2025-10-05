/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.common.value.qual.PolyValue;

// @skip-test until #153 is resolved.

    @Positive
public class Polymorphic4 {

    @Positive
  public static String @PolyValue [] quantify(String @PolyValue [] vars) {

    @Positive
    String[] result = new String[vars.length];

    @Positive
    return result;
    @Positive
  }
    @Positive
}
