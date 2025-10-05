// Source-based slice around line 175
// Method: <com.google.common.collect.RangeSet: void add(Range)>

   * range set for which both {@code a.enclosesAll(b)} and {@code a.encloses(range)}.
   *
   * <p>Note that {@code range} will be {@linkplain Range#span(Range) coalesced} with any ranges in
   * the range set that are {@linkplain Range#isConnected(Range) connected} with it. Moreover, if
   * {@code range} is empty, this is a no-op.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code add}
   *     operation
   */
  void add(Range<C> range);

  /**
   * Removes the specified range from this {@code RangeSet} (optional operation). After this
   * operation, if {@code range.contains(c)}, {@code this.contains(c)} will return {@code false}.
   *
   * <p>If {@code range} is empty, this is a no-op.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code remove}
   *     operation
   */
