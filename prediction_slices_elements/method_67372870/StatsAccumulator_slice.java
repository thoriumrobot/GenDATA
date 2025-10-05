// Source-based slice around line 264
// Method: <com.google.common.math.StatsAccumulator: double sum()>

   * <h3>Non-finite values</h3>
   *
   * <p>If the dataset contains {@link Double#NaN} then the result is {@link Double#NaN}. If it
   * contains both {@link Double#POSITIVE_INFINITY} and {@link Double#NEGATIVE_INFINITY} then the
   * result is {@link Double#NaN}. If it contains {@link Double#POSITIVE_INFINITY} and finite values
   * only or {@link Double#POSITIVE_INFINITY} only, the result is {@link Double#POSITIVE_INFINITY}.
   * If it contains {@link Double#NEGATIVE_INFINITY} and finite values only or {@link
   * Double#NEGATIVE_INFINITY} only, the result is {@link Double#NEGATIVE_INFINITY}.
   */
  public final double sum() {
    return mean * count;
  }

  /**
   * Returns the <a href="http://en.wikipedia.org/wiki/Variance#Population_variance">population
   * variance</a> of the values. The count must be non-zero.
   *
   * <p>This is guaranteed to return zero if the dataset contains only exactly one finite value. It
   * is not guaranteed to return zero when the dataset consists of the same value multiple times,
   * due to numerical errors. However, it is guaranteed never to return a negative result.
