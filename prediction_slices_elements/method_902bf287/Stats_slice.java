// Source-based slice around line 419
// Method: <com.google.common.math.Stats: boolean equals(Object)>

   * values in between the two calls, or if one is obtained from the other after round-tripping
   * through java serialization. However, floating point rounding errors mean that it may be false
   * for some instances where the statistics are mathematically equal, including instances
   * constructed from the same values in a different order... or (in the general case) even in the
   * same order. (It is guaranteed to return true for instances constructed from the same values in
   * the same order if {@code strictfp} is in effect, or if the system architecture guarantees
   * {@code strictfp}-like semantics.)
   */
  @Override
  public boolean equals(@Nullable Object obj) {
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    Stats other = (Stats) obj;
    return count == other.count
        && doubleToLongBits(mean) == doubleToLongBits(other.mean)
        && doubleToLongBits(sumOfSquaresOfDeltas) == doubleToLongBits(other.sumOfSquaresOfDeltas)
