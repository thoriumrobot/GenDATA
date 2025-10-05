// Source-based slice around line 355
// Method: <com.google.common.base.Splitter: Splitter trimResults(CharMatcher)>

   * or trailing characters matching the given {@code CharMatcher} from each returned substring. For
   * example, {@code Splitter.on(',').trimResults(CharMatcher.is('_')).split("_a ,_b_ ,c__")}
   * returns an iterable containing {@code ["a ", "b_ ", "c"]}.
   *
   * @param trimmer a {@link CharMatcher} that determines whether a character should be removed from
   *     the beginning/end of a subsequence
   * @return a splitter with the desired configuration
   */
  // TODO(kevinb): throw if a trimmer was already specified!
  public Splitter trimResults(CharMatcher trimmer) {
    checkNotNull(trimmer);
    return new Splitter(strategy, omitEmptyStrings, trimmer, limit);
  }

  /**
   * Splits {@code sequence} into string components and makes them available through an {@link
   * Iterator}, which may be lazily evaluated. If you want an eagerly computed {@link List}, use
   * {@link #splitToList(CharSequence)}. Java 8+ users may prefer {@link #splitToStream} instead.
   *
   * @param sequence the sequence of characters to split
