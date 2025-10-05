// Source-based slice around line 762
// Method: <com.google.common.base.CharMatcher: String trimFrom(CharSequence)>

   *
   * <p>Note that:
   *
   * {@snippet :
   * CharMatcher.inRange('\0', ' ').trimFrom(str)
   * }
   *
   * ... is equivalent to {@link String#trim()}.
   */
  public String trimFrom(CharSequence sequence) {
    int len = sequence.length();
    int first;
    int last;

    for (first = 0; first < len; first++) {
      if (!matches(sequence.charAt(first))) {
        break;
      }
    }
    for (last = len - 1; last > first; last--) {
