// Source-based slice around line 286
// Method: <com.google.common.primitives.UnsignedInts: int divide(int,int)>

   * Returns dividend / divisor, where the dividend and divisor are treated as unsigned 32-bit
   * quantities.
   *
   * <p><b>Java 8+ users:</b> use {@link Integer#divideUnsigned(int, int)} instead.
   *
   * @param dividend the dividend (numerator)
   * @param divisor the divisor (denominator)
   * @throws ArithmeticException if divisor is 0
   */
  public static int divide(int dividend, int divisor) {
    return (int) (toLong(dividend) / toLong(divisor));
  }

  /**
   * Returns dividend % divisor, where the dividend and divisor are treated as unsigned 32-bit
   * quantities.
   *
   * <p><b>Java 8+ users:</b> use {@link Integer#remainderUnsigned(int, int)} instead.
   *
   * @param dividend the dividend (numerator)
