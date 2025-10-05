// Source-based slice around line 437
// Method: <com.google.common.math.DoubleMath: double mean(int)>

   * <p>If these values are a sample drawn from a population, this is also an unbiased estimator of
   * the arithmetic mean of the population.
   *
   * @param values a nonempty series of values
   * @throws IllegalArgumentException if {@code values} is empty
   * @deprecated Use {@link Stats#meanOf} instead, noting the less strict handling of non-finite
   *     values.
   */
  @Deprecated
  public static double mean(int... values) {
    checkArgument(values.length > 0, "Cannot take mean of 0 values");
    // The upper bound on the length of an array and the bounds on the int values mean that, in
    // this case only, we can compute the sum as a long without risking overflow or loss of
    // precision. So we do that, as it's slightly quicker than the Knuth algorithm.
    long sum = 0;
    for (int index = 0; index < values.length; ++index) {
      sum += values[index];
    }
    return (double) sum / values.length;
  }
