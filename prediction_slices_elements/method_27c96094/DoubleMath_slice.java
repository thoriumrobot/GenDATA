// Source-based slice around line 226
// Method: <com.google.common.math.DoubleMath: double log2(double)>

   *   <li>If {@code x} is positive infinity, the result is positive infinity.
   *   <li>If {@code x} is positive or negative zero, the result is negative infinity.
   * </ul>
   *
   * <p>The computed result is within 1 ulp of the exact result.
   *
   * <p>If the result of this method will be immediately rounded to an {@code int}, {@link
   * #log2(double, RoundingMode)} is faster.
   */
  public static double log2(double x) {
    return log(x) / LN_2; // surprisingly within 1 ulp according to tests
  }

  /**
   * Returns the base 2 logarithm of a double value, rounded with the specified rounding mode to an
   * {@code int}.
   *
   * <p>Regardless of the rounding mode, this is faster than {@code (int) log2(x)}.
   *
   * @throws IllegalArgumentException if {@code x <= 0.0}, {@code x} is NaN, or {@code x} is
