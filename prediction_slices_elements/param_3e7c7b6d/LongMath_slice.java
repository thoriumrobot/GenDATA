// Source-based slice around line 306
// Method: <com.google.common.math.LongMath: long sqrt(long,RoundingMode)>


  /**
   * Returns the square root of {@code x}, rounded with the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x < 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code
   *     sqrt(x)} is not an integer
   */
  @GwtIncompatible // TODO
  public static long sqrt(long x, RoundingMode mode) {
    checkNonNegative("x", x);
    if (fitsInInt(x)) {
      return IntMath.sqrt((int) x, mode);
    }
    /*
     * Let k be the true value of floor(sqrt(x)), so that
     *
     *            k * k <= x          <  (k + 1) * (k + 1)
     * (double) (k * k) <= (double) x <= (double) ((k + 1) * (k + 1))
     *          since casting to double is nondecreasing.
