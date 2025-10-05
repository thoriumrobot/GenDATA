// Source-based slice around line 213
// Method: <com.google.common.base.Splitter: Splitter onPatternInternal(CommonPattern)>

   * @return a splitter, with default settings, that uses this pattern
   * @throws IllegalArgumentException if {@code separatorPattern} matches the empty string
   */
  @GwtIncompatible // java.util.regex
  public static Splitter on(Pattern separatorPattern) {
    return onPatternInternal(new JdkPattern(separatorPattern));
  }

  /** Internal utility; see {@link #on(Pattern)} instead. */
  static Splitter onPatternInternal(CommonPattern separatorPattern) {
    checkArgument(
        !separatorPattern.matcher("").matches(),
        "The pattern may not match the empty string: %s",
        separatorPattern);

    return new Splitter(
        (splitter, toSplit) -> {
          CommonMatcher matcher = separatorPattern.matcher(toSplit);
          return new SplittingIterator(splitter, toSplit) {
            @Override
