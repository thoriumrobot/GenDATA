// Source-based slice around line 251
// Method: <com.google.common.collect.RangeSet: void removeAll(Iterable)>

   * Removes all of the specified ranges from this range set (optional operation).
   *
   * <p>This is equivalent to calling {@link #remove} on each of the ranges in {@code other} in
   * turn.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code removeAll}
   *     operation
   * @since 21.0
   */
  default void removeAll(Iterable<Range<C>> ranges) {
    for (Range<C> range : ranges) {
      remove(range);
    }
  }

  // Object methods

  /**
   * Returns {@code true} if {@code obj} is another {@code RangeSet} that contains the same ranges
   * according to {@link Range#equals(Object)}.
