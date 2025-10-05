// Source-based slice around line 236
// Method: <com.google.common.math.PairedStats: int hashCode()>

  }

  /**
   * {@inheritDoc}
   *
   * <p><b>Note:</b> This hash code is consistent with exact equality of the calculated statistics,
   * including the floating point values. See the note on {@link #equals} for details.
   */
  @Override
  public int hashCode() {
    return Objects.hash(xStats, yStats, sumOfProductsOfDeltas);
  }

  @Override
  public String toString() {
    if (count() > 0) {
      return MoreObjects.toStringHelper(this)
          .add("xStats", xStats)
          .add("yStats", yStats)
          .add("populationCovariance", populationCovariance())
