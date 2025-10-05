// Source-based slice around line 244
// Method: <com.google.common.base.CharMatcher: CharMatcher javaLowerCase()>

  /**
   * Determines whether a BMP character is lower case according to {@linkplain
   * Character#isLowerCase(char) Java's definition}.
   *
   * @deprecated Some lowercase characters are supplementary characters; see the class
   *     documentation.
   * @since 19.0 (since 1.0 as constant {@code JAVA_LOWER_CASE})
   */
  @Deprecated
  public static CharMatcher javaLowerCase() {
    return JavaLowerCase.INSTANCE;
  }

  /**
   * Determines whether a character is an ISO control character as specified by {@link
   * Character#isISOControl(char)}.
   *
   * <p>All ISO control codes are on the BMP and thus supported by this API.
   *
   * @since 19.0 (since 1.0 as constant {@code JAVA_ISO_CONTROL})
