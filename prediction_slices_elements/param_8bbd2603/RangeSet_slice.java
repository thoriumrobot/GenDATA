// Source-based slice around line 239
// Method: <com.google.common.collect.RangeSet: void removeAll(RangeSet)>

   * operation). After this operation, if {@code other.contains(c)}, {@code this.contains(c)} will
   * return {@code false}.
   *
   * <p>This is equivalent to calling {@link #remove} on each of the ranges in {@code other} in
   * turn.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code removeAll}
   *     operation
   */
  void removeAll(RangeSet<C> other);

  /**
   * Removes all of the specified ranges from this range set (optional operation).
   *
   * <p>This is equivalent to calling {@link #remove} on each of the ranges in {@code other} in
   * turn.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code removeAll}
   *     operation
   * @since 21.0
