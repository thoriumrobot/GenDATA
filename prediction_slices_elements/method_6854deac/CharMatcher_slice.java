// Source-based slice around line 193
// Method: <com.google.common.base.CharMatcher: CharMatcher javaDigit()>

  /**
   * Determines whether a character is a BMP digit according to {@linkplain Character#isDigit(char)
   * Java's definition}. If you only care to match ASCII digits, you can use {@code inRange('0',
   * '9')}.
   *
   * @deprecated Many digits are supplementary characters; see the class documentation.
   * @since 19.0 (since 1.0 as constant {@code JAVA_DIGIT})
   */
  @Deprecated
  public static CharMatcher javaDigit() {
    return JavaDigit.INSTANCE;
  }

  /**
   * Determines whether a character is a BMP letter according to {@linkplain
   * Character#isLetter(char) Java's definition}. If you only care to match letters of the Latin
   * alphabet, you can use {@code inRange('a', 'z').or(inRange('A', 'Z'))}.
   *
   * @deprecated Most letters are supplementary characters; see the class documentation.
   * @since 19.0 (since 1.0 as constant {@code JAVA_LETTER})
