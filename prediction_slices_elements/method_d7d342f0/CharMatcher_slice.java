// Source-based slice around line 180
// Method: <com.google.common.base.CharMatcher: CharMatcher digit()>

  /**
   * Determines whether a character is a BMP digit according to <a
   * href="http://unicode.org/cldr/utility/list-unicodeset.jsp?a=%5Cp%7Bdigit%7D">Unicode</a>. If
   * you only care to match ASCII digits, you can use {@code inRange('0', '9')}.
   *
   * @deprecated Many digits are supplementary characters; see the class documentation.
   * @since 19.0 (since 1.0 as constant {@code DIGIT})
   */
  @Deprecated
  public static CharMatcher digit() {
    return Digit.INSTANCE;
  }

  /**
   * Determines whether a character is a BMP digit according to {@linkplain Character#isDigit(char)
   * Java's definition}. If you only care to match ASCII digits, you can use {@code inRange('0',
   * '9')}.
   *
   * @deprecated Many digits are supplementary characters; see the class documentation.
   * @since 19.0 (since 1.0 as constant {@code JAVA_DIGIT})
