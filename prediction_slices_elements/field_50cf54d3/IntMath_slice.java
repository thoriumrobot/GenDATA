// Source-based slice around line 631
// Method: com.google.common.math.IntMath.factorials

   * {@code n == 0}, or {@link Integer#MAX_VALUE} if the result does not fit in a {@code int}.
   *
   * @throws IllegalArgumentException if {@code n < 0}
   */
  public static int factorial(int n) {
    checkNonNegative("n", n);
    return (n < factorials.length) ? factorials[n] : Integer.MAX_VALUE;
  }

  private static final int[] factorials = {
    1,
    1,
    1 * 2,
    1 * 2 * 3,
    1 * 2 * 3 * 4,
    1 * 2 * 3 * 4 * 5,
    1 * 2 * 3 * 4 * 5 * 6,
    1 * 2 * 3 * 4 * 5 * 6 * 7,
    1 * 2 * 3 * 4 * 5 * 6 * 7 * 8,
    1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9,
