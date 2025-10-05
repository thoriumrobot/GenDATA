// Source-based slice around line 340
// Method: <com.google.common.base.Splitter: Splitter trimResults()>

  /**
   * Returns a splitter that behaves equivalently to {@code this} splitter, but automatically
   * removes leading and trailing {@linkplain CharMatcher#whitespace whitespace} from each returned
   * substring; equivalent to {@code trimResults(CharMatcher.whitespace())}. For example, {@code
   * Splitter.on(',').trimResults().split(" a, b ,c ")} returns an iterable containing {@code ["a",
   * "b", "c"]}.
   *
   * @return a splitter with the desired configuration
   */
  public Splitter trimResults() {
    return trimResults(CharMatcher.whitespace());
  }

  /**
   * Returns a splitter that behaves equivalently to {@code this} splitter, but removes all leading
   * or trailing characters matching the given {@code CharMatcher} from each returned substring. For
   * example, {@code Splitter.on(',').trimResults(CharMatcher.is('_')).split("_a ,_b_ ,c__")}
   * returns an iterable containing {@code ["a ", "b_ ", "c"]}.
   *
   * @param trimmer a {@link CharMatcher} that determines whether a character should be removed from
