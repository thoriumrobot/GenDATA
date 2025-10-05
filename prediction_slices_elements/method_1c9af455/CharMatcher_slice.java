// Source-based slice around line 218
// Method: <com.google.common.base.CharMatcher: CharMatcher javaLetterOrDigit()>


  /**
   * Determines whether a character is a BMP letter or digit according to {@linkplain
   * Character#isLetterOrDigit(char) Java's definition}.
   *
   * @deprecated Most letters and digits are supplementary characters; see the class documentation.
   * @since 19.0 (since 1.0 as constant {@code JAVA_LETTER_OR_DIGIT}).
   */
  @Deprecated
  public static CharMatcher javaLetterOrDigit() {
    return JavaLetterOrDigit.INSTANCE;
  }

  /**
   * Determines whether a BMP character is upper case according to {@linkplain
   * Character#isUpperCase(char) Java's definition}.
   *
   * @deprecated Some uppercase characters are supplementary characters; see the class
   *     documentation.
   * @since 19.0 (since 1.0 as constant {@code JAVA_UPPER_CASE})
