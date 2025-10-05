// Source-based slice around line 404
// Method: <com.google.common.math.StatsAccumulator: double calculateNewMeanNonFinite(double,double)>


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
     * the update formula can subtract infinities in cases where the naive formula would add them.
     *
     * Consequently:
     * 1. If the previous mean is finite and the new value is non-finite then the new mean is that
     *    value (whether it is NaN or infinity).
     * 2. If the new value is finite and the previous mean is non-finite then the mean is unchanged
     *    (whether it is NaN or infinity).
     * 3. If both the previous mean and the new value are non-finite and...
