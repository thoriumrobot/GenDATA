// Source-based slice around line 228
// Method: <com.google.common.collect.ContiguousSet: Range range()>

   */
  public abstract ContiguousSet<C> intersection(ContiguousSet<C> other);

  /**
   * Returns a range, closed on both ends, whose endpoints are the minimum and maximum values
   * contained in this set. This is equivalent to {@code range(CLOSED, CLOSED)}.
   *
   * @throws NoSuchElementException if this set is empty
   */
  public abstract Range<C> range();

  /**
   * Returns the minimal range with the given boundary types for which all values in this set are
   * {@linkplain Range#contains(Comparable) contained} within the range.
   *
   * <p>Note that this method will return ranges with unbounded endpoints if {@link BoundType#OPEN}
   * is requested for a domain minimum or maximum. For example, if {@code set} was created from the
   * range {@code [1..Integer.MAX_VALUE]} then {@code set.range(CLOSED, OPEN)} must return {@code
   * [1..∞)}.
   *
