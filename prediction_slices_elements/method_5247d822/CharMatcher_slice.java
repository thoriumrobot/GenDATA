// Source-based slice around line 618
// Method: <com.google.common.base.CharMatcher: String removeFrom(CharSequence)>

   * Returns a string containing all non-matching characters of a character sequence, in order. For
   * example:
   *
   * {@snippet :
   * CharMatcher.is('a').removeFrom("bazaar")
   * }
   *
   * ... returns {@code "bzr"}.
   */
  public String removeFrom(CharSequence sequence) {
    String string = sequence.toString();
    int pos = indexIn(string);
    if (pos == -1) {
      return string;
    }

    char[] chars = string.toCharArray();
    int spread = 1;

    // This unusual loop comes from extensive benchmarking
