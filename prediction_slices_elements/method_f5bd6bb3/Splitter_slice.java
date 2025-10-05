// Source-based slice around line 306
// Method: <com.google.common.base.Splitter: Splitter omitEmptyStrings()>

   * splitter always trims results first before checking for emptiness. So, for example, {@code
   * Splitter.on(':').omitEmptyStrings().trimResults().split(": : : ")} returns an empty iterable.
   *
   * <p>Note that it is ordinarily not possible for {@link #split(CharSequence)} to return an empty
   * iterable, but when using this option, it can (if the input sequence consists of nothing but
   * separators).
   *
   * @return a splitter with the desired configuration
   */
  public Splitter omitEmptyStrings() {
    return new Splitter(strategy, true, trimmer, limit);
  }

  /**
   * Returns a splitter that behaves equivalently to {@code this} splitter but stops splitting after
   * it reaches the limit. The limit defines the maximum number of items returned by the iterator,
   * or the maximum size of the list returned by {@link #splitToList}.
   *
   * <p>For example, {@code Splitter.on(',').limit(3).split("a,b,c,d")} returns an iterable
   * containing {@code ["a", "b", "c,d"]}. When omitting empty strings, the omitted strings do not
