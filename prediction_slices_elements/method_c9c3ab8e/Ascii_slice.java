// Source-based slice around line 479
// Method: <com.google.common.base.Ascii: String toUpperCase(CharSequence)>

  }

  /**
   * Returns a copy of the input character sequence in which all {@linkplain #isLowerCase(char)
   * lowercase ASCII characters} have been converted to uppercase. All other characters are copied
   * without modification.
   *
   * @since 14.0
   */
  public static String toUpperCase(CharSequence chars) {
    if (chars instanceof String) {
      return toUpperCase((String) chars);
    }
    char[] newChars = new char[chars.length()];
    for (int i = 0; i < newChars.length; i++) {
      newChars[i] = toUpperCase(chars.charAt(i));
    }
    return String.valueOf(newChars);
  }

