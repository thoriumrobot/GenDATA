// Source-based slice around line 166
// Method: <com.google.common.base.Splitter: Splitter on(String)>


  /**
   * Returns a splitter that uses the given fixed string as a separator. For example, {@code
   * Splitter.on(", ").split("foo, bar,baz")} returns an iterable containing {@code ["foo",
   * "bar,baz"]}.
   *
   * @param separator the literal, nonempty string to recognize as a separator
   * @return a splitter, with default settings, that recognizes that separator
   */
  public static Splitter on(String separator) {
    checkArgument(separator.length() != 0, "The separator may not be the empty string.");
    if (separator.length() == 1) {
      return Splitter.on(separator.charAt(0));
    }
    return new Splitter(
        (splitter, toSplit) ->
            new SplittingIterator(splitter, toSplit) {
              @Override
              public int separatorStart(int start) {
                int separatorLength = separator.length();
