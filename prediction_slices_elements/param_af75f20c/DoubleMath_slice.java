// Source-based slice around line 202
// Method: <com.google.common.math.DoubleMath: boolean isPowerOfTwo(double)>

    BigInteger result = BigInteger.valueOf(significand).shiftLeft(exponent - SIGNIFICAND_BITS);
    return (x < 0) ? result.negate() : result;
  }

  /**
   * Returns {@code true} if {@code x} is exactly equal to {@code 2^k} for some finite integer
   * {@code k}.
   */
  @GwtIncompatible // com.google.common.math.DoubleUtils
  public static boolean isPowerOfTwo(double x) {
    if (x > 0.0 && isFinite(x)) {
      long significand = getSignificand(x);
      return (significand & (significand - 1)) == 0;
    }
    return false;
  }

  /**
   * Returns the base 2 logarithm of a double value.
   *
