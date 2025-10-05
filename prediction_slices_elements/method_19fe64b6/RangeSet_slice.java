// Source-based slice around line 132
// Method: <com.google.common.collect.RangeSet: Set asRanges()>


  // Views

  /**
   * Returns a view of the {@linkplain Range#isConnected disconnected} ranges that make up this
   * range set. The returned set may be empty. The iterators returned by its {@link
   * Iterable#iterator} method return the ranges in increasing order of lower bound (equivalently,
   * of upper bound).
   */
  Set<Range<C>> asRanges();

  /**
   * Returns a descending view of the {@linkplain Range#isConnected disconnected} ranges that make
   * up this range set. The returned set may be empty. The iterators returned by its {@link
   * Iterable#iterator} method return the ranges in decreasing order of lower bound (equivalently,
   * of upper bound).
   *
   * @since 19.0
   */
  Set<Range<C>> asDescendingSetOfRanges();
