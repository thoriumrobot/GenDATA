// Source-based slice around line 314
// Method: <com.google.common.base.CharMatcher: CharMatcher anyOf(CharSequence)>

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
        return none();
      case 1:
        return is(sequence.charAt(0));
      case 2:
        return isEither(sequence.charAt(0), sequence.charAt(1));
      default:
        // TODO(lowasser): is it potentially worth just going ahead and building a precomputed
        // matcher?
