// Source-based slice around line 326
// Method: <com.google.common.base.Splitter: Splitter limit(int)>

   * count. Hence, {@code Splitter.on(',').limit(3).omitEmptyStrings().split("a,,,b,,,c,d")} returns
   * an iterable containing {@code ["a", "b", "c,d"]}. When trim is requested, all entries are
   * trimmed, including the last. Hence {@code Splitter.on(',').limit(3).trimResults().split(" a , b
   * , c , d ")} results in {@code ["a", "b", "c , d"]}.
   *
   * @param maxItems the maximum number of items returned
   * @return a splitter with the desired configuration
   * @since 9.0
   */
  public Splitter limit(int maxItems) {
    checkArgument(maxItems > 0, "must be greater than zero: %s", maxItems);
    return new Splitter(strategy, omitEmptyStrings, trimmer, maxItems);
  }

  /**
   * Returns a splitter that behaves equivalently to {@code this} splitter, but automatically
   * removes leading and trailing {@linkplain CharMatcher#whitespace whitespace} from each returned
   * substring; equivalent to {@code trimResults(CharMatcher.whitespace())}. For example, {@code
   * Splitter.on(',').trimResults().split(" a, b ,c ")} returns an iterable containing {@code ["a",
   * "b", "c"]}.
