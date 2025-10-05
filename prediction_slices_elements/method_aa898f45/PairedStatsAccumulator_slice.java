// Source-based slice around line 214
// Method: <com.google.common.math.PairedStatsAccumulator: LinearTransformation leastSquaresFit()>

   * <h3>Non-finite values</h3>
   *
   * <p>If the dataset contains any non-finite values ({@link Double#POSITIVE_INFINITY}, {@link
   * Double#NEGATIVE_INFINITY}, or {@link Double#NaN}) then the result is {@link
   * LinearTransformation#forNaN()}.
   *
   * @throws IllegalStateException if the dataset is empty or contains a single pair of values, or
   *     both the {@code x} and {@code y} dataset have zero population variance
   */
  public final LinearTransformation leastSquaresFit() {
    checkState(count() > 1);
    if (isNaN(sumOfProductsOfDeltas)) {
      return LinearTransformation.forNaN();
    }
    double xSumOfSquaresOfDeltas = xStats.sumOfSquaresOfDeltas();
    if (xSumOfSquaresOfDeltas > 0.0) {
      if (yStats.sumOfSquaresOfDeltas() > 0.0) {
        return LinearTransformation.mapping(xStats.mean(), yStats.mean())
            .withSlope(sumOfProductsOfDeltas / xSumOfSquaresOfDeltas);
      } else {
