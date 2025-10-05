// Source-based slice around line 441
// Method: <com.google.common.math.Stats: int hashCode()>

  }

  /**
   * {@inheritDoc}
   *
   * <p><b>Note:</b> This hash code is consistent with exact equality of the calculated statistics,
   * including the floating point values. See the note on {@link #equals} for details.
   */
  @Override
  public int hashCode() {
    return Objects.hash(count, mean, sumOfSquaresOfDeltas, min, max);
  }

  @Override
  public String toString() {
    if (count() > 0) {
      return MoreObjects.toStringHelper(this)
          .add("count", count)
          .add("mean", mean)
          .add("populationStandardDeviation", populationStandardDeviation())
