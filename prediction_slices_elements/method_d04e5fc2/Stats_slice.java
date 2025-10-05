// Source-based slice around line 337
// Method: <com.google.common.math.Stats: double sampleVariance()>

   * times, due to numerical errors. However, it is guaranteed never to return a negative result.
   *
   * <h3>Non-finite values</h3>
   *
   * <p>If the dataset contains any non-finite values ({@link Double#POSITIVE_INFINITY}, {@link
   * Double#NEGATIVE_INFINITY}, or {@link Double#NaN}) then the result is {@link Double#NaN}.
   *
   * @throws IllegalStateException if the dataset is empty or contains a single value
   */
  public double sampleVariance() {
    checkState(count > 1);
    if (isNaN(sumOfSquaresOfDeltas)) {
      return NaN;
    }
    return ensureNonNegative(sumOfSquaresOfDeltas) / (count - 1);
  }

  /**
   * Returns the <a
   * href="http://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation">
