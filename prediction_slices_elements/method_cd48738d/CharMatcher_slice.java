// Source-based slice around line 297
// Method: <com.google.common.base.CharMatcher: CharMatcher is(char)>

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

  /**
   * Returns a {@code char} matcher that matches any character except the BMP character specified.
   *
   * <p>To negate another {@code CharMatcher}, use {@link #negate()}.
   */
  public static CharMatcher isNot(char match) {
    return new IsNot(match);
