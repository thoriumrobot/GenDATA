// Source-based slice around line 222
// Method: <com.google.common.collect.RangeSet: void addAll(Iterable)>

   * this range set is the minimal range set that {@linkplain #enclosesAll(RangeSet) encloses} both
   * the original range set and each range in {@code other}.
   *
   * <p>This is equivalent to calling {@link #add} on each of the ranges in {@code other} in turn.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code addAll}
   *     operation
   * @since 21.0
   */
  default void addAll(Iterable<Range<C>> ranges) {
    for (Range<C> range : ranges) {
      add(range);
    }
  }

  /**
   * Removes all of the ranges from the specified range set from this range set (optional
   * operation). After this operation, if {@code other.contains(c)}, {@code this.contains(c)} will
   * return {@code false}.
   *
