// Source-based slice around line 791
// Method: <com.google.common.base.CharMatcher: String trimLeadingFrom(CharSequence)>

   * Returns a substring of the input character sequence that omits all matching BMP characters from
   * the beginning of the string. For example:
   *
   * {@snippet :
   * CharMatcher.anyOf("ab").trimLeadingFrom("abacatbab")
   * }
   *
   * ... returns {@code "catbab"}.
   */
  public String trimLeadingFrom(CharSequence sequence) {
    int len = sequence.length();
    for (int first = 0; first < len; first++) {
      if (!matches(sequence.charAt(first))) {
        return sequence.subSequence(first, len).toString();
      }
    }
    return "";
  }

  /**
