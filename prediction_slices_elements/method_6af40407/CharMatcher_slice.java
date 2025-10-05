// Source-based slice around line 290
// Method: <com.google.common.base.CharMatcher: CharMatcher singleWidth()>

   * <p><b>Note:</b> as the reference file evolves, we will modify this matcher to keep it up to
   * date.
   *
   * <p>See also <a href="http://www.unicode.org/reports/tr11/">UAX #11 East Asian Width</a>.
   *
   * @deprecated Many such characters are supplementary characters; see the class documentation.
   * @since 19.0 (since 1.0 as constant {@code SINGLE_WIDTH})
   */
  @Deprecated
  public static CharMatcher singleWidth() {
    return SingleWidth.INSTANCE;
  }

  // Static factories

  /** Returns a {@code char} matcher that matches only one specified BMP character. */
  public static CharMatcher is(char match) {
    return new Is(match);
  }

