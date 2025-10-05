// Source-based slice around line 140
// Method: <com.google.common.base.Splitter: Splitter on(CharMatcher)>

   * Returns a splitter that considers any single character matched by the given {@code CharMatcher}
   * to be a separator. For example, {@code
   * Splitter.on(CharMatcher.anyOf(";,")).split("foo,;bar,quux")} returns an iterable containing
   * {@code ["foo", "", "bar", "quux"]}.
   *
   * @param separatorMatcher a {@link CharMatcher} that determines whether a character is a
   *     separator
   * @return a splitter, with default settings, that uses this matcher
   */
  public static Splitter on(CharMatcher separatorMatcher) {
    checkNotNull(separatorMatcher);

    return new Splitter(
        (splitter, toSplit) ->
            new SplittingIterator(splitter, toSplit) {
              @Override
              int separatorStart(int start) {
                return separatorMatcher.indexIn(toSplit, start);
              }

