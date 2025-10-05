// Source-based slice around line 363
// Method: <com.google.common.math.BigIntegerMath: BigInteger divide(BigInteger,BigInteger,RoundingMode)>


  /**
   * Returns the result of dividing {@code p} by {@code q}, rounding using the specified {@code
   * RoundingMode}.
   *
   * @throws ArithmeticException if {@code q == 0}, or if {@code mode == UNNECESSARY} and {@code a}
   *     is not an integer multiple of {@code b}
   */
  @GwtIncompatible // TODO
  public static BigInteger divide(BigInteger p, BigInteger q, RoundingMode mode) {
    BigDecimal pDec = new BigDecimal(p);
    BigDecimal qDec = new BigDecimal(q);
    return pDec.divide(qDec, 0, mode).toBigIntegerExact();
  }

  /**
   * Returns {@code n!}, that is, the product of the first {@code n} positive integers, or {@code 1}
   * if {@code n == 0}.
   *
   * <p><b>Warning:</b> the result takes <i>O(n log n)</i> space, so use cautiously.
