// Source-based slice around line 494
// Method: <com.google.common.base.Ascii: char toUpperCase(char)>

      newChars[i] = toUpperCase(chars.charAt(i));
    }
    return String.valueOf(newChars);
  }

  /**
   * If the argument is a {@linkplain #isLowerCase(char) lowercase ASCII character}, returns the
   * uppercase equivalent. Otherwise returns the argument.
   */
  public static char toUpperCase(char c) {
    return isLowerCase(c) ? (char) (c ^ CASE_MASK) : c;
  }

  /**
   * Indicates whether {@code c} is one of the twenty-six lowercase ASCII alphabetic characters
   * between {@code 'a'} and {@code 'z'} inclusive. All others (including non-ASCII characters)
   * return {@code false}.
   */
  public static boolean isLowerCase(char c) {
    // Note: This was benchmarked against the alternate expression "(char)(c - 'a') < 26" (Nov '13)
