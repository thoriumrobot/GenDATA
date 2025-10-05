// Test case for issue #169: https://github.com/kelloggm/checker-framework/issues/169

// @skip-test until the issue is fixed

    @Positive
public class IndexOf {

    @Positive
  public static String m(String arg) {
    @Positive
    int split_pos = arg.indexOf(",-");
    @Positive
    if (split_pos == 0) {
      // Just discard the ',' if ",-" occurs at begining of string
    @Positive
      arg = arg.substring(1);
    @Positive
    }
    @Positive
    return arg;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
