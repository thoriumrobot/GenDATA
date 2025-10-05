// Source-based slice around line 264
// Method: <com.google.common.collect.RangeSet: boolean equals(Object)>

  }

  // Object methods

  /**
   * Returns {@code true} if {@code obj} is another {@code RangeSet} that contains the same ranges
   * according to {@link Range#equals(Object)}.
   */
  @Override
  boolean equals(@Nullable Object obj);

  /** Returns {@code asRanges().hashCode()}. */
  @Override
  int hashCode();

  /**
   * Returns a readable string representation of this range set. For example, if this {@code
   * RangeSet} consisted of {@code Range.closed(1, 3)} and {@code Range.greaterThan(4)}, this might
   * return {@code " [1..3](4..+∞)}"}.
   */
