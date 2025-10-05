// Source-based slice around line 389
// Method: <com.google.common.math.IntMath: int mod(int,int)>

   * mod(-1, 4) == 3
   * mod(-8, 4) == 0
   * mod(8, 4) == 0
   * }
   *
   * @throws ArithmeticException if {@code m <= 0}
   * @see <a href="http://docs.oracle.com/javase/specs/jls/se7/html/jls-15.html#jls-15.17.3">
   *     Remainder Operator</a>
   */
  public static int mod(int x, int m) {
    if (m <= 0) {
      throw new ArithmeticException("Modulus " + m + " must be > 0");
    }
    return Math.floorMod(x, m);
  }

  /**
   * Returns the greatest common divisor of {@code a, b}. Returns {@code 0} if {@code a == 0 && b ==
   * 0}.
   *
