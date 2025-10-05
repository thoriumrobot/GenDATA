// Source-based slice around line 626
// Method: <com.google.common.math.IntMath: int factorial(int)>


  @VisibleForTesting static final int FLOOR_SQRT_MAX_INT = 46340;

  /**
   * Returns {@code n!}, that is, the product of the first {@code n} positive integers, {@code 1} if
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
