// Source-based slice around line 305
// Method: <com.google.common.math.DoubleMath: double factorial(int)>

  /**
   * Returns {@code n!}, that is, the product of the first {@code n} positive integers, {@code 1} if
   * {@code n == 0}, or {@code n!}, or {@link Double#POSITIVE_INFINITY} if {@code n! >
   * Double.MAX_VALUE}.
   *
   * <p>The result is within 1 ulp of the true value.
   *
   * @throws IllegalArgumentException if {@code n < 0}
   */
  public static double factorial(int n) {
    checkNonNegative("n", n);
    if (n > MAX_FACTORIAL) {
      return Double.POSITIVE_INFINITY;
    } else {
      // Multiplying the last (n & 0xf) values into their own accumulator gives a more accurate
      // result than multiplying by everySixteenthFactorial[n >> 4] directly.
      double accum = 1.0;
      for (int i = 1 + (n & ~0xf); i <= n; i++) {
        accum *= i;
      }
