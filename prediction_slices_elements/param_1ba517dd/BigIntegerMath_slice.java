// Source-based slice around line 72
// Method: <com.google.common.math.BigIntegerMath: boolean isPowerOfTwo(BigInteger)>

   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @since 20.0
   */
  public static BigInteger floorPowerOfTwo(BigInteger x) {
    return BigInteger.ZERO.setBit(log2(x, FLOOR));
  }

  /** Returns {@code true} if {@code x} represents a power of two. */
  public static boolean isPowerOfTwo(BigInteger x) {
    checkNotNull(x);
    return x.signum() > 0 && x.getLowestSetBit() == x.bitLength() - 1;
  }

  /**
   * Returns the base-2 logarithm of {@code x}, rounded according to the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
   *     is not a power of two
