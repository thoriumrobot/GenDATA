// Source-based slice around line 714
// Method: <com.google.common.base.CharMatcher: String replaceFrom(CharSequence,CharSequence)>

   *
   * <p><b>Note:</b> If the replacement is a fixed string with only one character, you are better
   * off calling {@link #replaceFrom(CharSequence, char)} directly.
   *
   * @param sequence the character sequence to replace matching characters in
   * @param replacement the characters to append to the result string in place of each matching
   *     character in {@code sequence}
   * @return the new string
   */
  public String replaceFrom(CharSequence sequence, CharSequence replacement) {
    int replacementLen = replacement.length();
    if (replacementLen == 0) {
      return removeFrom(sequence);
    }
    if (replacementLen == 1) {
      return replaceFrom(sequence, replacement.charAt(0));
    }

    String string = sequence.toString();
    int pos = indexIn(string);
