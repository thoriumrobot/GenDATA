// Source-based slice around line 186
// Method: <com.google.common.math.DoubleMath: BigInteger roundToBigInteger(double,RoundingMode)>

   *       <li>{@code x} is infinite or NaN
   *       <li>{@code x} is not a mathematical integer and {@code mode} is {@link
   *           RoundingMode#UNNECESSARY}
   *     </ul>
   */
  // #roundIntermediate, java.lang.Math.getExponent, com.google.common.math.DoubleUtils
  @GwtIncompatible
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static BigInteger roundToBigInteger(double x, RoundingMode mode) {
    x = roundIntermediate(x, mode);
    if (MIN_LONG_AS_DOUBLE - x < 1.0 & x < MAX_LONG_AS_DOUBLE_PLUS_ONE) {
      return BigInteger.valueOf((long) x);
    }
    int exponent = getExponent(x);
    long significand = getSignificand(x);
    BigInteger result = BigInteger.valueOf(significand).shiftLeft(exponent - SIGNIFICAND_BITS);
    return (x < 0) ? result.negate() : result;
  }

