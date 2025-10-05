// Source-based slice around line 767
// Method: <com.google.common.math.LongMath: long factorial(int)>

  @VisibleForTesting static final long FLOOR_SQRT_MAX_LONG = 3037000499L;

  /**
   * Returns {@code n!}, that is, the product of the first {@code n} positive integers, {@code 1} if
   * {@code n == 0}, or {@link Long#MAX_VALUE} if the result does not fit in a {@code long}.
   *
   * @throws IllegalArgumentException if {@code n < 0}
   */
  @GwtIncompatible // TODO
  public static long factorial(int n) {
    checkNonNegative("n", n);
    return (n < factorials.length) ? factorials[n] : Long.MAX_VALUE;
  }

  static final long[] factorials = {
    1L,
    1L,
    1L * 2,
    1L * 2 * 3,
    1L * 2 * 3 * 4,
