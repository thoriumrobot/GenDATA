// Source-based slice around line 840
// Method: <com.google.common.base.CharMatcher: String collapseFrom(CharSequence,char)>

   * <p>The default implementation uses {@link #indexIn(CharSequence)} to find the first matching
   * character, then iterates the remainder of the sequence calling {@link #matches(char)} for each
   * character.
   *
   * @param sequence the character sequence to replace matching groups of characters in
   * @param replacement the character to append to the result string in place of each group of
   *     matching characters in {@code sequence}
   * @return the new string
   */
  public String collapseFrom(CharSequence sequence, char replacement) {
    // This implementation avoids unnecessary allocation.
    int len = sequence.length();
    for (int i = 0; i < len; i++) {
      char c = sequence.charAt(i);
      if (matches(c)) {
        if (c == replacement && (i == len - 1 || !matches(sequence.charAt(i + 1)))) {
          // a no-op replacement
          i++;
        } else {
          StringBuilder builder = new StringBuilder(len).append(sequence, 0, i).append(replacement);
