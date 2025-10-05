// Source-based slice around line 231
// Method: <com.google.common.base.CharMatcher: CharMatcher javaUpperCase()>

  /**
   * Determines whether a BMP character is upper case according to {@linkplain
   * Character#isUpperCase(char) Java's definition}.
   *
   * @deprecated Some uppercase characters are supplementary characters; see the class
   *     documentation.
   * @since 19.0 (since 1.0 as constant {@code JAVA_UPPER_CASE})
   */
  @Deprecated
  public static CharMatcher javaUpperCase() {
    return JavaUpperCase.INSTANCE;
  }

  /**
   * Determines whether a BMP character is lower case according to {@linkplain
   * Character#isLowerCase(char) Java's definition}.
   *
   * @deprecated Some lowercase characters are supplementary characters; see the class
   *     documentation.
   * @since 19.0 (since 1.0 as constant {@code JAVA_LOWER_CASE})
