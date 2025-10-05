// Source-based slice around line 680
// Method: <com.google.common.base.CharMatcher: String replaceFrom(CharSequence,char)>

   * <p>The default implementation uses {@link #indexIn(CharSequence)} to find the first matching
   * character, then iterates the remainder of the sequence calling {@link #matches(char)} for each
   * character.
   *
   * @param sequence the character sequence to replace matching characters in
   * @param replacement the character to append to the result string in place of each matching
   *     character in {@code sequence}
   * @return the new string
   */
  public String replaceFrom(CharSequence sequence, char replacement) {
    String string = sequence.toString();
    int pos = indexIn(string);
    if (pos == -1) {
      return string;
    }
    char[] chars = string.toCharArray();
    chars[pos] = replacement;
    for (int i = pos + 1; i < chars.length; i++) {
      if (matches(chars[i])) {
        chars[i] = replacement;
