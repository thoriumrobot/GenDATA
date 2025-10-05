// Tests string length refinement after startsWith or endsWith return true
// https://github.com/kelloggm/checker-framework/issues/56

    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class StartsEndsWith {

    @Positive
  final String prefix;

    @Positive
  StartsEndsWith(String prefix) {
    @Positive
    this.prefix = prefix;
    @Positive
  }

    @Positive
  String propertyName(String methodName) {
    @Positive
    if (methodName.startsWith(prefix)) {
    @Positive
      String result = methodName.substring(prefix.length());
    @Positive
      return result;
    @Positive
    } else {
    @Positive
      return null;
    @Positive
    }
    @Positive
  }

  // This particular test is here rather than in the framework tests because it depends on purity
  // annotations for these particular JDK methods.
    @Positive
  static void refineStartsConditional(String str, String prefix) {
    @Positive
    if (prefix.length() > 10 && str.startsWith(prefix)) {
    @Positive
      @MinLen(11) String s11 = str;
    @Positive
    }
    @Positive
  }
    @Positive
}

    @Positive
class StartsEndsWithExternal {
    @Positive
  public static final String staticFinalField = "str";
    @Positive
}

// CFWR semantic augmentation - variant 1
