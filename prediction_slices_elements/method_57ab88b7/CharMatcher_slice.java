// Source-based slice around line 272
// Method: <com.google.common.base.CharMatcher: CharMatcher invisible()>

   * PRIVATE_USE according to ICU4J.
   *
   * <p>See also the Unicode Default_Ignorable_Code_Point property (available via ICU).
   *
   * @deprecated Most invisible characters are supplementary characters; see the class
   *     documentation.
   * @since 19.0 (since 1.0 as constant {@code INVISIBLE})
   */
  @Deprecated
  public static CharMatcher invisible() {
    return Invisible.INSTANCE;
  }

  /**
   * Determines whether a character is single-width (not double-width). When in doubt, this matcher
   * errs on the side of returning {@code false} (that is, it tends to assume a character is
   * double-width).
   *
   * <p><b>Note:</b> as the reference file evolves, we will modify this matcher to keep it up to
   * date.
