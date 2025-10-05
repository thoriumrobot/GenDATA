// Source-based slice around line 158
// Method: <com.google.common.base.CharMatcher: CharMatcher breakingWhitespace()>

  }

  /**
   * Determines whether a character is a breaking whitespace (that is, a whitespace which can be
   * interpreted as a break between words for formatting purposes). See {@link #whitespace()} for a
   * discussion of that term.
   *
   * @since 19.0 (since 2.0 as constant {@code BREAKING_WHITESPACE})
   */
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
