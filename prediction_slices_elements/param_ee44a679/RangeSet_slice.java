// Source-based slice around line 186
// Method: <com.google.common.collect.RangeSet: void remove(Range)>

  /**
   * Removes the specified range from this {@code RangeSet} (optional operation). After this
   * operation, if {@code range.contains(c)}, {@code this.contains(c)} will return {@code false}.
   *
   * <p>If {@code range} is empty, this is a no-op.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code remove}
   *     operation
   */
  void remove(Range<C> range);

  /**
   * Removes all ranges from this {@code RangeSet} (optional operation). After this operation,
   * {@code this.contains(c)} will return false for all {@code c}.
   *
   * <p>This is equivalent to {@code remove(Range.all())}.
   *
   * @throws UnsupportedOperationException if this range set does not support the {@code clear}
   *     operation
   */
