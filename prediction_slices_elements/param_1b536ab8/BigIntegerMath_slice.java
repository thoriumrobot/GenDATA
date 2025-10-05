// Source-based slice around line 220
// Method: <com.google.common.math.BigIntegerMath: BigInteger sqrt(BigInteger,RoundingMode)>

  /**
   * Returns the square root of {@code x}, rounded with the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x < 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code
   *     sqrt(x)} is not an integer
   */
  @GwtIncompatible // TODO
  @SuppressWarnings("fallthrough")
  public static BigInteger sqrt(BigInteger x, RoundingMode mode) {
    checkNonNegative("x", x);
    if (fitsInLong(x)) {
      return BigInteger.valueOf(LongMath.sqrt(x.longValue(), mode));
    }
    BigInteger sqrtFloor = sqrtFloor(x);
    switch (mode) {
      case UNNECESSARY:
        checkRoundingUnnecessary(sqrtFloor.pow(2).equals(x)); // fall through
      case FLOOR:
      case DOWN:
