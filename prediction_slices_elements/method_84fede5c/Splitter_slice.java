// Source-based slice around line 271
// Method: <com.google.common.base.Splitter: Splitter fixedLength(int)>

   * <p><b>Exception:</b> for consistency with separator-based splitters, {@code split("")} does not
   * yield an empty iterable, but an iterable containing {@code ""}. This is the only case in which
   * {@code Iterables.size(split(input))} does not equal {@code IntMath.divide(input.length(),
   * length, CEILING)}. To avoid this behavior, use {@code omitEmptyStrings}.
   *
   * @param length the desired length of pieces after splitting, a positive integer
   * @return a splitter, with default settings, that can split into fixed sized pieces
   * @throws IllegalArgumentException if {@code length} is zero or negative
   */
  public static Splitter fixedLength(int length) {
    checkArgument(length > 0, "The length may not be less than 1");

    return new Splitter(
        (splitter, toSplit) ->
            new SplittingIterator(splitter, toSplit) {
              @Override
              public int separatorStart(int start) {
                int nextChunkStart = start + length;
                return (nextChunkStart < toSplit.length() ? nextChunkStart : -1);
              }
