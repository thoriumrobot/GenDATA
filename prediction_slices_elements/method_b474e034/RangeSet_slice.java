// Source-based slice around line 75
// Method: <com.google.common.collect.RangeSet: boolean intersects(Range)>

  @Nullable Range<C> rangeContaining(C value);

  /**
   * Returns {@code true} if there exists a non-empty range enclosed by both a member range in this
   * range set and the specified range. This is equivalent to calling {@code
   * subRangeSet(otherRange)} and testing whether the resulting range set is non-empty.
   *
   * @since 20.0
   */
  boolean intersects(Range<C> otherRange);

  /**
   * Returns {@code true} if there exists a member range in this range set which {@linkplain
   * Range#encloses encloses} the specified range.
   */
  boolean encloses(Range<C> otherRange);

  /**
   * Returns {@code true} if for each member range in {@code other} there exists a member range in
   * this range set which {@linkplain Range#encloses encloses} it. It follows that {@code
