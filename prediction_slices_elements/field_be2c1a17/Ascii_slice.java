// Source-based slice around line 400
// Method: com.google.common.base.Ascii.CASE_MASK


  /**
   * The maximum value of an ASCII character.
   *
   * @since 9.0 (was type {@code int} before 12.0)
   */
  public static final char MAX = 127;

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
