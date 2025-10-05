// Source-based slice around line 208
// Method: <com.google.common.base.Splitter: Splitter on(Pattern)>

   * For example, {@code Splitter.on(Pattern.compile("\r?\n")).split(entireFile)} splits a string
   * into lines whether it uses DOS-style or UNIX-style line terminators.
   *
   * @param separatorPattern the pattern that determines whether a subsequence is a separator. This
   *     pattern may not match the empty string.
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

