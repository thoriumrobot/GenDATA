// Source-based slice around line 63
// Method: <com.google.common.math.LongMath: long ceilingPowerOfTwo(long)>

  /**
   * Returns the smallest power of two greater than or equal to {@code x}. This is equivalent to
   * {@code checkedPow(2, log2(x, CEILING))}.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException of the next-higher power of two is not representable as a {@code
   *     long}, i.e. when {@code x > 2^62}
   * @since 20.0
   */
  public static long ceilingPowerOfTwo(long x) {
    checkPositive("x", x);
    if (x > MAX_SIGNED_POWER_OF_TWO) {
      throw new ArithmeticException("ceilingPowerOfTwo(" + x + ") is not representable as a long");
    }
    return 1L << -Long.numberOfLeadingZeros(x - 1);
  }

  /**
   * Returns the largest power of two less than or equal to {@code x}. This is equivalent to {@code
   * checkedPow(2, log2(x, FLOOR))}.
