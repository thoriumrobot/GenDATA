// Source-based slice around line 147
// Method: <com.google.common.base.CharMatcher: CharMatcher whitespace()>

   * of several definitions of "whitespace"</a>.)
   *
   * <p>All Unicode White_Space characters are on the BMP and thus supported by this API.
   *
   * <p><b>Note:</b> as the Unicode definition evolves, we will modify this matcher to keep it up to
   * date.
   *
   * @since 19.0 (since 1.0 as constant {@code WHITESPACE})
   */
  public static CharMatcher whitespace() {
    return Whitespace.INSTANCE;
  }

  /**
   * Determines whether a character is a breaking whitespace (that is, a whitespace which can be
   * interpreted as a break between words for formatting purposes). See {@link #whitespace()} for a
   * discussion of that term.
   *
   * @since 19.0 (since 2.0 as constant {@code BREAKING_WHITESPACE})
   */
