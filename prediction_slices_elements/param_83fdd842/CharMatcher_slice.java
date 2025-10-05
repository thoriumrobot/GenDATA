// Source-based slice around line 811
// Method: <com.google.common.base.CharMatcher: String trimTrailingFrom(CharSequence)>

   * Returns a substring of the input character sequence that omits all matching BMP characters from
   * the end of the string. For example:
   *
   * {@snippet :
   * CharMatcher.anyOf("ab").trimTrailingFrom("abacatbab")
   * }
   *
   * ... returns {@code "abacat"}.
   */
  public String trimTrailingFrom(CharSequence sequence) {
    int len = sequence.length();
    for (int last = len - 1; last >= 0; last--) {
      if (!matches(sequence.charAt(last))) {
        return sequence.subSequence(0, last + 1).toString();
      }
    }
    return "";
  }

  /**
