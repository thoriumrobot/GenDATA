/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Tests for index annotations on string methods in the annotated JDK

    @Positive
public class StringMethods {

    @Positive
  void testCharAt(String s, int i) {
    // ::  error: (argument)
    @Positive
    s.charAt(i);
    // ::  error: (argument)
    @Positive
    s.codePointAt(i);

    @Positive
    if (i >= 0 && i < s.length()) {
    @Positive
      s.charAt(i);
    @Positive
      s.codePointAt(i);
    @Positive
    }
    @Positive
  }

    @Positive
  void testCodePointBefore(String s) {
    // ::  error: (argument)
    @Positive
    s.codePointBefore(0);

    @Positive
    if (s.length() > 0) {
    @Positive
      s.codePointBefore(s.length());
    @Positive
    }
    @Positive
  }

    @Positive
  void testSubstring(String s) {
    @Positive
    s.substring(0);
    @Positive
    s.substring(0, 0);
    @Positive
    s.substring(s.length());
    @Positive
    s.substring(s.length(), s.length());
    @Positive
    s.substring(0, s.length());
    // ::  error: (argument)
    @Positive
    s.substring(1);
    // ::  error: (argument)
    @Positive
    s.substring(0, 1);
    @Positive
  }

    @Positive
  void testIndexOf(String s, char c) {
    @Positive
    int i = s.indexOf(c);
    @Positive
    if (i != -1) {
    @Positive
      s.charAt(i);
    @Positive
    }
    @Positive
  }
    @Positive
}
