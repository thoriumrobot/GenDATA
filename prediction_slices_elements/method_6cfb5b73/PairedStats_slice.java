// Source-based slice around line 216
// Method: <com.google.common.math.PairedStats: boolean equals(Object)>

   * adding any values in between the two calls, or if one is obtained from the other after
   * round-tripping through java serialization. However, floating point rounding errors mean that it
   * may be false for some instances where the statistics are mathematically equal, including
   * instances constructed from the same values in a different order... or (in the general case)
   * even in the same order. (It is guaranteed to return true for instances constructed from the
   * same values in the same order if {@code strictfp} is in effect, or if the system architecture
   * guarantees {@code strictfp}-like semantics.)
   */
  @Override
  public boolean equals(@Nullable Object obj) {
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    PairedStats other = (PairedStats) obj;
    return xStats.equals(other.xStats)
        && yStats.equals(other.yStats)
        && doubleToLongBits(sumOfProductsOfDeltas) == doubleToLongBits(other.sumOfProductsOfDeltas);
