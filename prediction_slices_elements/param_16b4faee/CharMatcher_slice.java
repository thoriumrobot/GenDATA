// Source-based slice around line 950
// Method: <com.google.common.base.CharMatcher: String showCharacter(char)>

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
    for (int i = 0; i < 4; i++) {
      tmp[5 - i] = hex.charAt(c & 0xF);
      c = (char) (c >> 4);
    }
    return String.copyValueOf(tmp);
  }

  // Fast matchers
