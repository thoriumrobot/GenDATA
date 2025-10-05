// Source-based slice around line 772
// Method: com.google.common.math.LongMath.factorials

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
    1L * 2 * 3 * 4 * 5,
    1L * 2 * 3 * 4 * 5 * 6,
    1L * 2 * 3 * 4 * 5 * 6 * 7,
    1L * 2 * 3 * 4 * 5 * 6 * 7 * 8,
    1L * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9,
