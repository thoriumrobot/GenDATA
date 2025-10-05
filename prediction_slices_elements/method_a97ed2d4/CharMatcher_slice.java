// Source-based slice around line 206
// Method: <com.google.common.base.CharMatcher: CharMatcher javaLetter()>

  /**
   * Determines whether a character is a BMP letter according to {@linkplain
   * Character#isLetter(char) Java's definition}. If you only care to match letters of the Latin
   * alphabet, you can use {@code inRange('a', 'z').or(inRange('A', 'Z'))}.
   *
   * @deprecated Most letters are supplementary characters; see the class documentation.
   * @since 19.0 (since 1.0 as constant {@code JAVA_LETTER})
   */
  @Deprecated
  public static CharMatcher javaLetter() {
    return JavaLetter.INSTANCE;
  }

  /**
   * Determines whether a character is a BMP letter or digit according to {@linkplain
   * Character#isLetterOrDigit(char) Java's definition}.
   *
   * @deprecated Most letters and digits are supplementary characters; see the class documentation.
   * @since 19.0 (since 1.0 as constant {@code JAVA_LETTER_OR_DIGIT}).
   */
