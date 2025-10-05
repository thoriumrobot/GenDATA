// Source-based slice around line 122
// Method: <com.google.common.collect.RangeSet: Range span()>

  /** Returns {@code true} if this range set contains no ranges. */
  boolean isEmpty();

  /**
   * Returns the minimal range which {@linkplain Range#encloses(Range) encloses} all ranges in this
   * range set.
   *
   * @throws NoSuchElementException if this range set is {@linkplain #isEmpty() empty}
   */
  Range<C> span();

  // Views

  /**
   * Returns a view of the {@linkplain Range#isConnected disconnected} ranges that make up this
   * range set. The returned set may be empty. The iterators returned by its {@link
   * Iterable#iterator} method return the ranges in increasing order of lower bound (equivalently,
   * of upper bound).
   */
  Set<Range<C>> asRanges();
