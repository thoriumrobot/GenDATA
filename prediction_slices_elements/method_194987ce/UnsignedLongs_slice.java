// Source-based slice around line 250
// Method: <com.google.common.primitives.UnsignedLongs: long divide(long,long)>

   * Returns dividend / divisor, where the dividend and divisor are treated as unsigned 64-bit
   * quantities.
   *
   * <p><b>Java 8+ users:</b> use {@link Long#divideUnsigned(long, long)} instead.
   *
   * @param dividend the dividend (numerator)
   * @param divisor the divisor (denominator)
   * @throws ArithmeticException if divisor is 0
   */
  public static long divide(long dividend, long divisor) {
    if (divisor < 0) { // i.e., divisor >= 2^63:
      if (compare(dividend, divisor) < 0) {
        return 0; // dividend < divisor
      } else {
        return 1; // dividend >= divisor
      }
    }

    // Optimization - use signed division if dividend < 2^63
    if (dividend >= 0) {
