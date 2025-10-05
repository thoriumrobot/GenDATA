// Source-based slice around line 463
// Method: <com.google.common.math.DoubleMath: double mean(long)>

   * the arithmetic mean of the population.
   *
   * @param values a nonempty series of values, which will be converted to {@code double} values
   *     (this may cause loss of precision for longs of magnitude over 2^53 (slightly over 9e15))
   * @throws IllegalArgumentException if {@code values} is empty
   * @deprecated Use {@link Stats#meanOf} instead, noting the less strict handling of non-finite
   *     values.
   */
  @Deprecated
  public static double mean(long... values) {
    checkArgument(values.length > 0, "Cannot take mean of 0 values");
    long count = 1;
    double mean = values[0];
    for (int index = 1; index < values.length; ++index) {
      count++;
      // Art of Computer Programming vol. 2, Knuth, 4.2.2, (15)
      mean += (values[index] - mean) / count;
    }
    return mean;
  }
