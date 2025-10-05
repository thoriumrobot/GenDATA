// Source-based slice around line 474
// Method: <com.google.common.math.Stats: double meanOf(Iterable)>

   * Returns the <a href="http://en.wikipedia.org/wiki/Arithmetic_mean">arithmetic mean</a> of the
   * values. The count must be non-zero.
   *
   * <p>The definition of the mean is the same as {@link Stats#mean}.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
   *     cause loss of precision)
   * @throws IllegalArgumentException if the dataset is empty
   */
  public static double meanOf(Iterable<? extends Number> values) {
    return meanOf(values.iterator());
  }

  /**
   * Returns the <a href="http://en.wikipedia.org/wiki/Arithmetic_mean">arithmetic mean</a> of the
   * values. The count must be non-zero.
   *
   * <p>The definition of the mean is the same as {@link Stats#mean}.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
