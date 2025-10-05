// Source-based slice around line 306
// Method: <com.google.common.base.CharMatcher: CharMatcher isNot(char)>

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
  }

  /**
   * Returns a {@code char} matcher that matches any BMP character present in the given character
   * sequence. Returns a bogus matcher if the sequence contains supplementary characters.
   */
  public static CharMatcher anyOf(CharSequence sequence) {
    switch (sequence.length()) {
      case 0:
