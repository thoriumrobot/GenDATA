// Source-based slice around line 511
// Method: <com.google.common.math.DoubleMath: double mean(Iterator)>

   * @param values a nonempty series of values, which will be converted to {@code double} values
   *     (this may cause loss of precision)
   * @throws IllegalArgumentException if {@code values} is empty or contains any non-finite value
   * @deprecated Use {@link Stats#meanOf} instead, noting the less strict handling of non-finite
   *     values.
   */
  @Deprecated
  // com.google.common.math.DoubleUtils
  @GwtIncompatible
  public static double mean(Iterator<? extends Number> values) {
    checkArgument(values.hasNext(), "Cannot take mean of 0 values");
    long count = 1;
    double mean = checkFinite(values.next().doubleValue());
    while (values.hasNext()) {
      double value = checkFinite(values.next().doubleValue());
      count++;
      // Art of Computer Programming vol. 2, Knuth, 4.2.2, (15)
      mean += (value - mean) / count;
    }
    return mean;
