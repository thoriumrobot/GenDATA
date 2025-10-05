// Source-based slice around line 396
// Method: <com.google.common.math.StatsAccumulator: double sumOfSquaresOfDeltas()>

   * Double#NEGATIVE_INFINITY} only then the result is {@link Double#NEGATIVE_INFINITY}.
   *
   * @throws IllegalStateException if the dataset is empty
   */
  public double max() {
    checkState(count != 0);
    return max;
  }

  double sumOfSquaresOfDeltas() {
    return sumOfSquaresOfDeltas;
  }

  /**
   * Calculates the new value for the accumulated mean when a value is added, in the case where at
   * least one of the previous mean and the value is non-finite.
   */
  static double calculateNewMeanNonFinite(double previousMean, double value) {
    /*
     * Desired behaviour is to match the results of applying the naive mean formula. In particular,
