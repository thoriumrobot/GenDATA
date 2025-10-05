// Source-based slice around line 167
// Method: <com.google.common.base.CharMatcher: CharMatcher ascii()>

  public static CharMatcher breakingWhitespace() {
    return BreakingWhitespace.INSTANCE;
  }

  /**
   * Determines whether a character is ASCII, meaning that its code point is less than 128.
   *
   * @since 19.0 (since 1.0 as constant {@code ASCII})
   */
  public static CharMatcher ascii() {
    return Ascii.INSTANCE;
  }

  /**
   * Determines whether a character is a BMP digit according to <a
   * href="http://unicode.org/cldr/utility/list-unicodeset.jsp?a=%5Cp%7Bdigit%7D">Unicode</a>. If
   * you only care to match ASCII digits, you can use {@code inRange('0', '9')}.
   *
   * @deprecated Many digits are supplementary characters; see the class documentation.
   * @since 19.0 (since 1.0 as constant {@code DIGIT})
