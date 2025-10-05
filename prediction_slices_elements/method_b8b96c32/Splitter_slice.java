// Source-based slice around line 249
// Method: <com.google.common.base.Splitter: Splitter onPattern(String)>

   * to {@code Splitter.on(Pattern.compile(pattern))}.
   *
   * @param separatorPattern the pattern that determines whether a subsequence is a separator. This
   *     pattern may not match the empty string.
   * @return a splitter, with default settings, that uses this pattern
   * @throws IllegalArgumentException if {@code separatorPattern} matches the empty string or is a
   *     malformed expression
   */
  @GwtIncompatible // java.util.regex
  public static Splitter onPattern(String separatorPattern) {
    return onPatternInternal(Platform.compilePattern(separatorPattern));
  }

  /**
   * Returns a splitter that divides strings into pieces of the given length. For example, {@code
   * Splitter.fixedLength(2).split("abcde")} returns an iterable containing {@code ["ab", "cd",
   * "e"]}. The last piece can be smaller than {@code length} but will never be empty.
   *
   * <p><b>Note:</b> if {@link #fixedLength} is used in conjunction with {@link #limit}, the final
   * split piece <i>may be longer than the specified fixed length</i>. This is because the splitter
