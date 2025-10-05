// Source-based slice around line 610
// Method: <com.google.common.collect.Range: Range span(Range)>

   * other}. For example, the span of {@code [1..3]} and {@code (5..7)} is {@code [1..7)}.
   *
   * <p><i>If</i> the input ranges are {@linkplain #isConnected connected}, the returned range can
   * also be called their <i>union</i>. If they are not, note that the span might contain values
   * that are not contained in either input range.
   *
   * <p>Like {@link #intersection(Range) intersection}, this operation is commutative, associative
   * and idempotent. Unlike it, it is always well-defined for any two input ranges.
   */
  public Range<C> span(Range<C> other) {
    int lowerCmp = lowerBound.compareTo(other.lowerBound);
    int upperCmp = upperBound.compareTo(other.upperBound);
    if (lowerCmp <= 0 && upperCmp >= 0) {
      return this;
    } else if (lowerCmp >= 0 && upperCmp <= 0) {
      return other;
    } else {
      Cut<C> newLower = (lowerCmp <= 0) ? lowerBound : other.lowerBound;
      Cut<C> newUpper = (upperCmp >= 0) ? upperBound : other.upperBound;
      return create(newLower, newUpper);
