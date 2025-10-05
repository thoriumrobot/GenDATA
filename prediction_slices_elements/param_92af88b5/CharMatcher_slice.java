// Source-based slice around line 864
// Method: <com.google.common.base.CharMatcher: String trimAndCollapseFrom(CharSequence,char)>

    // no replacement needed
    return sequence.toString();
  }

  /**
   * Collapses groups of matching characters exactly as {@link #collapseFrom} does, except that
   * groups of matching BMP characters at the start or end of the sequence are removed without
   * replacement.
   */
  public String trimAndCollapseFrom(CharSequence sequence, char replacement) {
    // This implementation avoids unnecessary allocation.
    int len = sequence.length();
    int first = 0;
    int last = len - 1;

    while (first < len && matches(sequence.charAt(first))) {
      first++;
    }

    while (last > first && matches(sequence.charAt(last))) {
