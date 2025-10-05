// Source-based slice around line 475
// Method: <com.google.common.math.LongMath: long mod(long,long)>

   * mod(-8, 4) == 0
   * mod(8, 4) == 0
   * }
   *
   * @throws ArithmeticException if {@code m <= 0}
   * @see <a href="http://docs.oracle.com/javase/specs/jls/se7/html/jls-15.html#jls-15.17.3">
   *     Remainder Operator</a>
   */
  @GwtIncompatible // TODO
  public static long mod(long x, long m) {
    if (m <= 0) {
      throw new ArithmeticException("Modulus must be positive");
    }
    return Math.floorMod(x, m);
  }

  /**
   * Returns the greatest common divisor of {@code a, b}. Returns {@code 0} if {@code a == 0 && b ==
   * 0}.
   *
