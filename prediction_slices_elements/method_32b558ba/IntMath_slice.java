// Source-based slice around line 726
// Method: <com.google.common.math.IntMath: boolean isPrime(int)>

   * Returns {@code false} if {@code n} is zero, one, or a composite number (one which <i>can</i> be
   * factored into smaller positive integers).
   *
   * <p>To test larger numbers, use {@link LongMath#isPrime} or {@link BigInteger#isProbablePrime}.
   *
   * @throws IllegalArgumentException if {@code n} is negative
   * @since 20.0
   */
  @GwtIncompatible // TODO
  public static boolean isPrime(int n) {
    return LongMath.isPrime(n);
  }

  /**
   * Returns the closest representable {@code int} to the absolute value of {@code x}.
   *
   * <p>This is the same thing as the true absolute value of {@code x} except in the case when
   * {@code x} is {@link Integer#MIN_VALUE}, in which case this returns {@link Integer#MAX_VALUE}.
   * (Note that {@code Integer.MAX_VALUE} is mathematically equal to {@code -Integer.MIN_VALUE -
   * 1}.)
