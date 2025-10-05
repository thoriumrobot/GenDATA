// Source-based slice around line 283
// Method: <com.google.common.math.StatsAccumulator: double populationVariance()>

   * due to numerical errors. However, it is guaranteed never to return a negative result.
   *
   * <h3>Non-finite values</h3>
   *
   * <p>If the dataset contains any non-finite values ({@link Double#POSITIVE_INFINITY}, {@link
   * Double#NEGATIVE_INFINITY}, or {@link Double#NaN}) then the result is {@link Double#NaN}.
   *
   * @throws IllegalStateException if the dataset is empty
   */
  public final double populationVariance() {
    checkState(count != 0);
    if (isNaN(sumOfSquaresOfDeltas)) {
      return NaN;
    }
    if (count == 1) {
      return 0.0;
    }
    return ensureNonNegative(sumOfSquaresOfDeltas) / count;
  }

