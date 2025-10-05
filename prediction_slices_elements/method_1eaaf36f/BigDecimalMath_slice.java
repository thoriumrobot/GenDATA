// Source-based slice around line 54
// Method: <com.google.common.math.BigDecimalMath: double roundToDouble(BigDecimal,RoundingMode)>

   * default rounding mode: if the two nearest representable values are equally near, the one with
   * the least significant bit zero is chosen. (In such cases, both of the nearest representable
   * values are even integers; this method returns the one that is a multiple of a greater power of
   * two.)
   *
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
   *     is not precisely representable as a {@code double}
   * @since 30.0
   */
  public static double roundToDouble(BigDecimal x, RoundingMode mode) {
    return BigDecimalToDoubleRounder.INSTANCE.roundToDouble(x, mode);
  }

  private static final class BigDecimalToDoubleRounder extends ToDoubleRounder<BigDecimal> {
    static final BigDecimalToDoubleRounder INSTANCE = new BigDecimalToDoubleRounder();

    private BigDecimalToDoubleRounder() {}

    @Override
    double roundToDoubleArbitrarily(BigDecimal bigDecimal) {
