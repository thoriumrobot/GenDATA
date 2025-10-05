// Source-based slice around line 455
// Method: <com.google.common.base.Ascii: String toUpperCase(String)>

  public static char toLowerCase(char c) {
    return isUpperCase(c) ? (char) (c ^ CASE_MASK) : c;
  }

  /**
   * Returns a copy of the input string in which all {@linkplain #isLowerCase(char) lowercase ASCII
   * characters} have been converted to uppercase. All other characters are copied without
   * modification.
   */
  public static String toUpperCase(String string) {
    int length = string.length();
    for (int i = 0; i < length; i++) {
      if (isLowerCase(string.charAt(i))) {
        char[] chars = string.toCharArray();
        for (; i < length; i++) {
          char c = chars[i];
          if (isLowerCase(c)) {
            chars[i] = (char) (c ^ CASE_MASK);
          }
        }
