// Source-based slice around line 167
// Method: <com.google.common.math.PairedStatsAccumulator: double pearsonsCorrelationCoefficient()>

   *
   * <h3>Non-finite values</h3>
   *
   * <p>If the dataset contains any non-finite values ({@link Double#POSITIVE_INFINITY}, {@link
   * Double#NEGATIVE_INFINITY}, or {@link Double#NaN}) then the result is {@link Double#NaN}.
   *
   * @throws IllegalStateException if the dataset is empty or contains a single pair of values, or
   *     either the {@code x} and {@code y} dataset has zero population variance
   */
  public final double pearsonsCorrelationCoefficient() {
    checkState(count() > 1);
    if (isNaN(sumOfProductsOfDeltas)) {
      return NaN;
    }
    double xSumOfSquaresOfDeltas = xStats.sumOfSquaresOfDeltas();
    double ySumOfSquaresOfDeltas = yStats.sumOfSquaresOfDeltas();
    checkState(xSumOfSquaresOfDeltas > 0.0);
    checkState(ySumOfSquaresOfDeltas > 0.0);
    // The product of two positive numbers can be zero if the multiplication underflowed. We
    // force a positive value by effectively rounding up to MIN_VALUE.
