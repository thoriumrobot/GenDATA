// Source-based slice around line 491
// Method: <com.google.common.math.DoubleMath: double mean(Iterable)>

   * @param values a nonempty series of values, which will be converted to {@code double} values
   *     (this may cause loss of precision)
   * @throws IllegalArgumentException if {@code values} is empty or contains any non-finite value
   * @deprecated Use {@link Stats#meanOf} instead, noting the less strict handling of non-finite
   *     values.
   */
  @Deprecated
  // com.google.common.math.DoubleUtils
  @GwtIncompatible
  public static double mean(Iterable<? extends Number> values) {
    return mean(values.iterator());
  }

  /**
   * Returns the <a href="http://en.wikipedia.org/wiki/Arithmetic_mean">arithmetic mean</a> of
   * {@code values}.
   *
   * <p>If these values are a sample drawn from a population, this is also an unbiased estimator of
   * the arithmetic mean of the population.
   *
