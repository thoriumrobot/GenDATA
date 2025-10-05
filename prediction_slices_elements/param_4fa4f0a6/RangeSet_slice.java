// Source-based slice around line 209
// Method: <com.google.common.collect.RangeSet: void addAll(RangeSet)>

   * Adds all of the ranges from the specified range set to this range set (optional operation).
   * After this operation, this range set is the minimal range set that {@linkplain
   * #enclosesAll(RangeSet) encloses} both the original range set and {@code other}.
   *
   * <p>This is equivalent to calling {@link #add} on each of the ranges in {@code other} in turn.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code addAll}
   *     operation
   */
  void addAll(RangeSet<C> other);

  /**
   * Adds all of the specified ranges to this range set (optional operation). After this operation,
   * this range set is the minimal range set that {@linkplain #enclosesAll(RangeSet) encloses} both
   * the original range set and each range in {@code other}.
   *
   * <p>This is equivalent to calling {@link #add} on each of the ranges in {@code other} in turn.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code addAll}
   *     operation
