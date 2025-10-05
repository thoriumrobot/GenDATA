// Source-based slice around line 300
// Method: <com.google.common.primitives.UnsignedInts: int remainder(int,int)>

   * Returns dividend % divisor, where the dividend and divisor are treated as unsigned 32-bit
   * quantities.
   *
   * <p><b>Java 8+ users:</b> use {@link Integer#remainderUnsigned(int, int)} instead.
   *
   * @param dividend the dividend (numerator)
   * @param divisor the divisor (denominator)
   * @throws ArithmeticException if divisor is 0
   */
  public static int remainder(int dividend, int divisor) {
    return (int) (toLong(dividend) % toLong(divisor));
  }

  /**
   * Returns the unsigned {@code int} value represented by the given string.
   *
   * <p>Accepts a decimal, hexadecimal, or octal number given by specifying the following prefix:
   *
   * <ul>
   *   <li>{@code 0x}<i>HexDigits</i>
