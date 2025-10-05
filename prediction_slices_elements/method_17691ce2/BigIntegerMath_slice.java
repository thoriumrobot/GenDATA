// Source-based slice around line 67
// Method: <com.google.common.math.BigIntegerMath: BigInteger floorPowerOfTwo(BigInteger)>

  }

  /**
   * Returns the largest power of two less than or equal to {@code x}. This is equivalent to {@code
   * BigInteger.valueOf(2).pow(log2(x, FLOOR))}.
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
