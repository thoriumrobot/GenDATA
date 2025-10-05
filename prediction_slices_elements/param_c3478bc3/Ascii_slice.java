// Source-based slice around line 446
// Method: <com.google.common.base.Ascii: char toLowerCase(char)>

      newChars[i] = toLowerCase(chars.charAt(i));
    }
    return String.valueOf(newChars);
  }

  /**
   * If the argument is an {@linkplain #isUpperCase(char) uppercase ASCII character}, returns the
   * lowercase equivalent. Otherwise returns the argument.
   */
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
