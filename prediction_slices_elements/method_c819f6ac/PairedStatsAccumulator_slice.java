// Source-based slice around line 146
// Method: <com.google.common.math.PairedStatsAccumulator: double sampleCovariance()>

   * multiple times, due to numerical errors.
   *
   * <h3>Non-finite values</h3>
   *
   * <p>If the dataset contains any non-finite values ({@link Double#POSITIVE_INFINITY}, {@link
   * Double#NEGATIVE_INFINITY}, or {@link Double#NaN}) then the result is {@link Double#NaN}.
   *
   * @throws IllegalStateException if the dataset is empty or contains a single pair of values
   */
  public final double sampleCovariance() {
    checkState(count() > 1);
    return sumOfProductsOfDeltas / (count() - 1);
  }

  /**
   * Returns the <a href="http://mathworld.wolfram.com/CorrelationCoefficient.html">Pearson's or
   * product-moment correlation coefficient</a> of the values. The count must greater than one, and
   * the {@code x} and {@code y} values must both have non-zero population variance (i.e. {@code
   * xStats().populationVariance() > 0.0 && yStats().populationVariance() > 0.0}). The result is not
   * guaranteed to be exactly +/-1 even when the data are perfectly (anti-)correlated, due to
