// Source-based slice around line 431
// Method: <com.google.common.base.Ascii: String toLowerCase(CharSequence)>

  }

  /**
   * Returns a copy of the input character sequence in which all {@linkplain #isUpperCase(char)
   * uppercase ASCII characters} have been converted to lowercase. All other characters are copied
   * without modification.
   *
   * @since 14.0
   */
  public static String toLowerCase(CharSequence chars) {
    if (chars instanceof String) {
      return toLowerCase((String) chars);
    }
    char[] newChars = new char[chars.length()];
    for (int i = 0; i < newChars.length; i++) {
      newChars[i] = toLowerCase(chars.charAt(i));
    }
    return String.valueOf(newChars);
  }

