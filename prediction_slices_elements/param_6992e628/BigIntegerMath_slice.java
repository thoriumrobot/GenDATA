// Source-based slice around line 143
// Method: <com.google.common.math.BigIntegerMath: int log10(BigInteger,RoundingMode)>

  /**
   * Returns the base-10 logarithm of {@code x}, rounded according to the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
   *     is not a power of ten
   */
  @GwtIncompatible // TODO
  @SuppressWarnings("fallthrough")
  public static int log10(BigInteger x, RoundingMode mode) {
    checkPositive("x", x);
    if (fitsInLong(x)) {
      return LongMath.log10(x.longValue(), mode);
    }

    int approxLog10 = (int) (log2(x, FLOOR) * LN_2 / LN_10);
    BigInteger approxPow = BigInteger.TEN.pow(approxLog10);
    int approxCmp = approxPow.compareTo(x);

    /*
