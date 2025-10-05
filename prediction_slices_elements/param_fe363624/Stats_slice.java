// Source-based slice around line 563
// Method: <com.google.common.math.Stats: double meanOf(long)>

   * Returns the <a href="http://en.wikipedia.org/wiki/Arithmetic_mean">arithmetic mean</a> of the
   * values. The count must be non-zero.
   *
   * <p>The definition of the mean is the same as {@link Stats#mean}.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision for longs of magnitude over 2^53 (slightly over 9e15))
   * @throws IllegalArgumentException if the dataset is empty
   */
  public static double meanOf(long... values) {
    checkArgument(values.length > 0);
    double mean = values[0];
    for (int index = 1; index < values.length; index++) {
      double value = values[index];
      if (isFinite(value) && isFinite(mean)) {
        // Art of Computer Programming vol. 2, Knuth, 4.2.2, (15)
        mean += (value - mean) / (index + 1);
      } else {
        mean = calculateNewMeanNonFinite(mean, value);
      }
