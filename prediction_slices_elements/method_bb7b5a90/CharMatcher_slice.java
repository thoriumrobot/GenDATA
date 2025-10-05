// Source-based slice around line 942
// Method: <com.google.common.base.CharMatcher: String toString()>

  public boolean test(Character character) {
    return matches(character);
  }

  /**
   * Returns a string representation of this {@code CharMatcher}, such as {@code
   * CharMatcher.or(WHITESPACE, JAVA_DIGIT)}.
   */
  @Override
  public String toString() {
    return super.toString();
  }

  /**
   * Returns the Java Unicode escape sequence for the given {@code char}, in the form "\u12AB" where
   * "12AB" is the four hexadecimal digits representing the 16-bit code unit.
   */
  private static String showCharacter(char c) {
    String hex = "0123456789ABCDEF";
    char[] tmp = {'\\', 'u', '\0', '\0', '\0', '\0'};
