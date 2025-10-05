// Source-based slice around line 197
// Method: <com.google.common.collect.RangeSet: void clear()>

  /**
   * Removes all ranges from this {@code RangeSet} (optional operation). After this operation,
   * {@code this.contains(c)} will return false for all {@code c}.
   *
   * <p>This is equivalent to {@code remove(Range.all())}.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code clear}
   *     operation
   */
  void clear();

  /**
   * Adds all of the ranges from the specified range set to this range set (optional operation).
   * After this operation, this range set is the minimal range set that {@linkplain
   * #enclosesAll(RangeSet) encloses} both the original range set and {@code other}.
   *
   * <p>This is equivalent to calling {@link #add} on each of the ranges in {@code other} in turn.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code addAll}
   *     operation
