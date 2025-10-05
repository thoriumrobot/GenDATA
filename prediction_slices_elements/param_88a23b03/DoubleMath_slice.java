// Source-based slice around line 411
// Method: <com.google.common.math.DoubleMath: double mean(double)>

   *
   * @param values a nonempty series of values
   * @throws IllegalArgumentException if {@code values} is empty or contains any non-finite value
   * @deprecated Use {@link Stats#meanOf} instead, noting the less strict handling of non-finite
   *     values.
   */
  @Deprecated
  // com.google.common.math.DoubleUtils
  @GwtIncompatible
  public static double mean(double... values) {
    checkArgument(values.length > 0, "Cannot take mean of 0 values");
    long count = 1;
    double mean = checkFinite(values[0]);
    for (int index = 1; index < values.length; ++index) {
      checkFinite(values[index]);
      count++;
      // Art of Computer Programming vol. 2, Knuth, 4.2.2, (15)
      mean += (values[index] - mean) / count;
    }
    return mean;
