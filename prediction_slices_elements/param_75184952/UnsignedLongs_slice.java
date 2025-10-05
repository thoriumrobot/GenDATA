// Source-based slice around line 286
// Method: <com.google.common.primitives.UnsignedLongs: long remainder(long,long)>

   * quantities.
   *
   * <p><b>Java 8+ users:</b> use {@link Long#remainderUnsigned(long, long)} instead.
   *
   * @param dividend the dividend (numerator)
   * @param divisor the divisor (denominator)
   * @throws ArithmeticException if divisor is 0
   * @since 11.0
   */
  public static long remainder(long dividend, long divisor) {
    if (divisor < 0) { // i.e., divisor >= 2^63:
      if (compare(dividend, divisor) < 0) {
        return dividend; // dividend < divisor
      } else {
        return dividend - divisor; // dividend >= divisor
      }
    }

    // Optimization - use signed modulus if dividend < 2^63
    if (dividend >= 0) {
