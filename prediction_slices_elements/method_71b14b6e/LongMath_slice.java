// Source-based slice around line 1238
// Method: <com.google.common.math.LongMath: double roundToDouble(long,RoundingMode)>

   * the least significant bit zero is chosen. (In such cases, both of the nearest representable
   * values are even integers; this method returns the one that is a multiple of a greater power of
   * two.)
   *
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
   *     is not precisely representable as a {@code double}
   * @since 30.0
   */
  @GwtIncompatible
  public static double roundToDouble(long x, RoundingMode mode) {
    // Logic adapted from ToDoubleRounder.
    double roundArbitrarily = (double) x;
    long roundArbitrarilyAsLong = (long) roundArbitrarily;
    int cmpXToRoundArbitrarily;

    if (roundArbitrarilyAsLong == Long.MAX_VALUE) {
      /*
       * For most values, the conversion from roundArbitrarily to roundArbitrarilyAsLong is
       * lossless. In that case we can compare x to roundArbitrarily using Long.compare(x,
       * roundArbitrarilyAsLong). The exception is for values where the conversion to double rounds
