// Source-based slice around line 128
// Method: <com.google.common.base.CharMatcher: CharMatcher none()>

  public static CharMatcher any() {
    return Any.INSTANCE;
  }

  /**
   * Matches no characters.
   *
   * @since 19.0 (since 1.0 as constant {@code NONE})
   */
  public static CharMatcher none() {
    return None.INSTANCE;
  }

  /**
   * Determines whether a character is whitespace according to the latest Unicode standard, as
   * illustrated <a
   * href="http://unicode.org/cldr/utility/list-unicodeset.jsp?a=%5Cp%7Bwhitespace%7D">here</a>.
   * This is not the same definition used by other Java APIs. (See a <a
   * href="https://docs.google.com/spreadsheets/d/1kq4ECwPjHX9B8QUCTPclgsDCXYaj7T-FlT4tB5q3ahk/edit">comparison
   * of several definitions of "whitespace"</a>.)
