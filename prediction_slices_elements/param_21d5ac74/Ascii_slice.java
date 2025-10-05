// Source-based slice around line 514
// Method: <com.google.common.base.Ascii: boolean isUpperCase(char)>

    // and found to perform at least as well, or better.
    return (c >= 'a') && (c <= 'z');
  }

  /**
   * Indicates whether {@code c} is one of the twenty-six uppercase ASCII alphabetic characters
   * between {@code 'A'} and {@code 'Z'} inclusive. All others (including non-ASCII characters)
   * return {@code false}.
   */
  public static boolean isUpperCase(char c) {
    return (c >= 'A') && (c <= 'Z');
  }

  /**
   * Truncates the given character sequence to the given maximum length. If the length of the
   * sequence is greater than {@code maxLength}, the returned string will be exactly {@code
   * maxLength} chars in length and will end with the given {@code truncationIndicator}. Otherwise,
   * the sequence will be returned as a string with no changes to the content.
   *
   * <p>Examples:
