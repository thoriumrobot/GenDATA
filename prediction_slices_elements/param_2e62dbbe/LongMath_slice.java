// Source-based slice around line 451
// Method: <com.google.common.math.LongMath: int mod(long,int)>

   * mod(-8, 4) == 0
   * mod(8, 4) == 0
   * }
   *
   * @throws ArithmeticException if {@code m <= 0}
   * @see <a href="http://docs.oracle.com/javase/specs/jls/se7/html/jls-15.html#jls-15.17.3">
   *     Remainder Operator</a>
   */
  @GwtIncompatible // TODO
  public static int mod(long x, int m) {
    // Cast is safe because the result is guaranteed in the range [0, m)
    return (int) mod(x, (long) m);
  }

  /**
   * Returns {@code x mod m}, a non-negative value less than {@code m}. This differs from {@code x %
   * m}, which might be negative.
   *
   * <p>For example:
   *
