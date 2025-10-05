// Source-based slice around line 256
// Method: <com.google.common.base.CharMatcher: CharMatcher javaIsoControl()>


  /**
   * Determines whether a character is an ISO control character as specified by {@link
   * Character#isISOControl(char)}.
   *
   * <p>All ISO control codes are on the BMP and thus supported by this API.
   *
   * @since 19.0 (since 1.0 as constant {@code JAVA_ISO_CONTROL})
   */
  public static CharMatcher javaIsoControl() {
    return JavaIsoControl.INSTANCE;
  }

  /**
   * Determines whether a character is invisible; that is, if its Unicode category is any of
   * SPACE_SEPARATOR, LINE_SEPARATOR, PARAGRAPH_SEPARATOR, CONTROL, FORMAT, SURROGATE, and
   * PRIVATE_USE according to ICU4J.
   *
   * <p>See also the Unicode Default_Ignorable_Code_Point property (available via ICU).
   *
