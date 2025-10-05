// Source-based slice around line 276
// Method: <com.google.common.collect.RangeSet: String toString()>

  @Override
  int hashCode();

  /**
   * Returns a readable string representation of this range set. For example, if this {@code
   * RangeSet} consisted of {@code Range.closed(1, 3)} and {@code Range.greaterThan(4)}, this might
   * return {@code " [1..3](4..+∞)}"}.
   */
  @Override
  String toString();
}
