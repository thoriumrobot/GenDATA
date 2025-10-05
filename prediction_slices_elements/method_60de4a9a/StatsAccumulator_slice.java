// Source-based slice around line 247
// Method: <com.google.common.math.StatsAccumulator: double mean()>

   * <p>If the dataset contains {@link Double#NaN} then the result is {@link Double#NaN}. If it
   * contains both {@link Double#POSITIVE_INFINITY} and {@link Double#NEGATIVE_INFINITY} then the
   * result is {@link Double#NaN}. If it contains {@link Double#POSITIVE_INFINITY} and finite values
   * only or {@link Double#POSITIVE_INFINITY} only, the result is {@link Double#POSITIVE_INFINITY}.
   * If it contains {@link Double#NEGATIVE_INFINITY} and finite values only or {@link
   * Double#NEGATIVE_INFINITY} only, the result is {@link Double#NEGATIVE_INFINITY}.
   *
   * @throws IllegalStateException if the dataset is empty
   */
  public double mean() {
    checkState(count != 0);
    return mean;
  }

  /**
   * Returns the sum of the values.
   *
   * <h3>Non-finite values</h3>
   *
   * <p>If the dataset contains {@link Double#NaN} then the result is {@link Double#NaN}. If it
