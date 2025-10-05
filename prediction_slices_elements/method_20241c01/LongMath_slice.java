// Source-based slice around line 996
// Method: <com.google.common.math.LongMath: boolean isPrime(long)>

   * Returns {@code false} if {@code n} is zero, one, or a composite number (one which <i>can</i> be
   * factored into smaller positive integers).
   *
   * <p>To test larger numbers, use {@link BigInteger#isProbablePrime}.
   *
   * @throws IllegalArgumentException if {@code n} is negative
   * @since 20.0
   */
  @GwtIncompatible // TODO
  public static boolean isPrime(long n) {
    if (n < 2) {
      checkNonNegative("n", n);
      return false;
    }
    if (n < 66) {
      // Encode all primes less than 66 into mask without 0 and 1.
      long mask =
          (1L << (2 - 2))
              | (1L << (3 - 2))
              | (1L << (5 - 2))
