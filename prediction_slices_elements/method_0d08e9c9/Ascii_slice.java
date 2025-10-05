// Source-based slice around line 407
// Method: <com.google.common.base.Ascii: String toLowerCase(String)>


  /** A bit mask which selects the bit encoding ASCII character case. */
  private static final char CASE_MASK = 0x20;

  /**
   * Returns a copy of the input string in which all {@linkplain #isUpperCase(char) uppercase ASCII
   * characters} have been converted to lowercase. All other characters are copied without
   * modification.
   */
  public static String toLowerCase(String string) {
    int length = string.length();
    for (int i = 0; i < length; i++) {
      if (isUpperCase(string.charAt(i))) {
        char[] chars = string.toCharArray();
        for (; i < length; i++) {
          char c = chars[i];
          if (isUpperCase(c)) {
            chars[i] = (char) (c ^ CASE_MASK);
          }
        }
