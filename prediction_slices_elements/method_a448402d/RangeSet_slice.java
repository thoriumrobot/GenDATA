// Source-based slice around line 142
// Method: <com.google.common.collect.RangeSet: Set asDescendingSetOfRanges()>


  /**
   * Returns a descending view of the {@linkplain Range#isConnected disconnected} ranges that make
   * up this range set. The returned set may be empty. The iterators returned by its {@link
   * Iterable#iterator} method return the ranges in decreasing order of lower bound (equivalently,
   * of upper bound).
   *
   * @since 19.0
   */
  Set<Range<C>> asDescendingSetOfRanges();

  /**
   * Returns a view of the complement of this {@code RangeSet}.
   *
   * <p>The returned view supports the {@link #add} operation if this {@code RangeSet} supports
   * {@link #remove}, and vice versa.
   */
  RangeSet<C> complement();

  /**
