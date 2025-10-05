// Source-based slice around line 536
// Method: <com.google.common.collect.Range: Range intersection(Range)>

   *
   * <p>The intersection exists if and only if the two ranges are {@linkplain #isConnected
   * connected}.
   *
   * <p>The intersection operation is commutative, associative and idempotent, and its identity
   * element is {@link Range#all}).
   *
   * @throws IllegalArgumentException if {@code isConnected(connectedRange)} is {@code false}
   */
  public Range<C> intersection(Range<C> connectedRange) {
    int lowerCmp = lowerBound.compareTo(connectedRange.lowerBound);
    int upperCmp = upperBound.compareTo(connectedRange.upperBound);
    if (lowerCmp >= 0 && upperCmp <= 0) {
      return this;
    } else if (lowerCmp <= 0 && upperCmp >= 0) {
      return connectedRange;
    } else {
      Cut<C> newLower = (lowerCmp >= 0) ? lowerBound : connectedRange.lowerBound;
      Cut<C> newUpper = (upperCmp <= 0) ? upperBound : connectedRange.upperBound;

