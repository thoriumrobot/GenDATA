// Source-based slice around line 373
// Method: <com.google.common.math.StatsAccumulator: double min()>

   *
   * <p>If the dataset contains {@link Double#NaN} then the result is {@link Double#NaN}. If it
   * contains {@link Double#NEGATIVE_INFINITY} and not {@link Double#NaN} then the result is {@link
   * Double#NEGATIVE_INFINITY}. If it contains {@link Double#POSITIVE_INFINITY} and finite values
   * only then the result is the lowest finite value. If it contains {@link
   * Double#POSITIVE_INFINITY} only then the result is {@link Double#POSITIVE_INFINITY}.
   *
   * @throws IllegalStateException if the dataset is empty
   */
  public double min() {
    checkState(count != 0);
    return min;
  }

  /**
   * Returns the highest value in the dataset. The count must be non-zero.
   *
   * <h3>Non-finite values</h3>
   *
   * <p>If the dataset contains {@link Double#NaN} then the result is {@link Double#NaN}. If it
