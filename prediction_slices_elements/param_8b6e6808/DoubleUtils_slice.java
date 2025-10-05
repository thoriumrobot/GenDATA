// Source-based slice around line 88
// Method: <com.google.common.math.DoubleUtils: double bigToDouble(BigInteger)>

  /*
   * Returns x scaled by a power of 2 such that it is in the range [1, 2). Assumes x is positive,
   * normal, and finite.
   */
  static double scaleNormalize(double x) {
    long significand = doubleToRawLongBits(x) & SIGNIFICAND_MASK;
    return longBitsToDouble(significand | ONE_BITS);
  }

  static double bigToDouble(BigInteger x) {
    // This is an extremely fast implementation of BigInteger.doubleValue(). JDK patch pending.
    BigInteger absX = x.abs();
    int exponent = absX.bitLength() - 1;
    // exponent == floor(log2(abs(x)))
    if (exponent < Long.SIZE - 1) {
      return x.longValue();
    } else if (exponent > MAX_EXPONENT) {
      return x.signum() * POSITIVE_INFINITY;
    }

