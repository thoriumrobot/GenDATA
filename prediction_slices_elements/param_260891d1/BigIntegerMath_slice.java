// Source-based slice around line 324
// Method: <com.google.common.math.BigIntegerMath: double roundToDouble(BigInteger,RoundingMode)>

   * the least significant bit zero is chosen. (In such cases, both of the nearest representable
   * values are even integers; this method returns the one that is a multiple of a greater power of
   * two.)
   *
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
   *     is not precisely representable as a {@code double}
   * @since 30.0
   */
  @GwtIncompatible
  public static double roundToDouble(BigInteger x, RoundingMode mode) {
    return BigIntegerToDoubleRounder.INSTANCE.roundToDouble(x, mode);
  }

  @GwtIncompatible
  private static final class BigIntegerToDoubleRounder extends ToDoubleRounder<BigInteger> {
    static final BigIntegerToDoubleRounder INSTANCE = new BigIntegerToDoubleRounder();

    private BigIntegerToDoubleRounder() {}

    @Override
