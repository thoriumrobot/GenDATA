// Source-based slice around line 95
// Method: <com.google.common.math.PairedStats: double populationCovariance()>

   * times, due to numerical errors.
   *
   * <h3>Non-finite values</h3>
   *
   * <p>If the dataset contains any non-finite values ({@link Double#POSITIVE_INFINITY}, {@link
   * Double#NEGATIVE_INFINITY}, or {@link Double#NaN}) then the result is {@link Double#NaN}.
   *
   * @throws IllegalStateException if the dataset is empty
   */
  public double populationCovariance() {
    checkState(count() != 0);
    return sumOfProductsOfDeltas / count();
  }

  /**
   * Returns the sample covariance of the values. The count must be greater than one.
   *
   * <p>This is not guaranteed to return zero when the dataset consists of the same pair of values
   * multiple times, due to numerical errors.
   *
