/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue #168: https://github.com/kelloggm/checker-framework/issues/168

    @Positive
public class EndsWith2 {

    @Positive
  public static String invertBrackets(String classname) {

    // Get the array depth (if any)
    @Positive
    int array_depth = 0;
    @Positive
    String brackets = "";
    @Positive
    while (classname.endsWith("[]")) {
    @Positive
      brackets = brackets + classname.substring(classname.length() - 2);
    @Positive
      classname = classname.substring(0, classname.length() - 2);
    @Positive
      array_depth++;
    @Positive
    }
    @Positive
    return brackets + classname;
    @Positive
  }
    @Positive
}
