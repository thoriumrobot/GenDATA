// Source-based slice around line 317
// Method: <com.google.common.math.Stats: double populationStandardDeviation()>

   * due to numerical errors. However, it is guaranteed never to return a negative result.
   *
   * <h3>Non-finite values</h3>
   *
   * <p>If the dataset contains any non-finite values ({@link Double#POSITIVE_INFINITY}, {@link
   * Double#NEGATIVE_INFINITY}, or {@link Double#NaN}) then the result is {@link Double#NaN}.
   *
   * @throws IllegalStateException if the dataset is empty
   */
  public double populationStandardDeviation() {
    return Math.sqrt(populationVariance());
  }

  /**
   * Returns the <a href="http://en.wikipedia.org/wiki/Variance#Sample_variance">unbiased sample
   * variance</a> of the values. If this dataset is a sample drawn from a population, this is an
   * unbiased estimator of the population variance of the population. The count must be greater than
   * one.
   *
   * <p>This is not guaranteed to return zero when the dataset consists of the same value multiple
