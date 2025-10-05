// Source-based slice around line 488
// Method: <com.google.common.math.Stats: double meanOf(Iterator)>

   * Returns the <a href="http://en.wikipedia.org/wiki/Arithmetic_mean">arithmetic mean</a> of the
   * values. The count must be non-zero.
   *
   * <p>The definition of the mean is the same as {@link Stats#mean}.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision)
   * @throws IllegalArgumentException if the dataset is empty
   */
  public static double meanOf(Iterator<? extends Number> values) {
    checkArgument(values.hasNext());
    long count = 1;
    double mean = values.next().doubleValue();
    while (values.hasNext()) {
      double value = values.next().doubleValue();
      count++;
      if (isFinite(value) && isFinite(mean)) {
        // Art of Computer Programming vol. 2, Knuth, 4.2.2, (15)
        mean += (value - mean) / count;
      } else {
