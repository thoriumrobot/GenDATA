// Source-based slice around line 66
// Method: <com.google.common.collect.RangeSet: Range rangeContaining(C)>

  // Query methods

  /** Determines whether any of this range set's member ranges contains {@code value}. */
  boolean contains(C value);

  /**
   * Returns the unique range from this range set that {@linkplain Range#contains contains} {@code
   * value}, or {@code null} if this range set does not contain {@code value}.
   */
  @Nullable Range<C> rangeContaining(C value);

  /**
   * Returns {@code true} if there exists a non-empty range enclosed by both a member range in this
   * range set and the specified range. This is equivalent to calling {@code
   * subRangeSet(otherRange)} and testing whether the resulting range set is non-empty.
   *
   * @since 20.0
   */
  boolean intersects(Range<C> otherRange);

