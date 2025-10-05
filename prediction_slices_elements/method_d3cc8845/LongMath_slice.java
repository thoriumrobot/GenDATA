// Source-based slice around line 579
// Method: <com.google.common.math.LongMath: long checkedMultiply(long,long)>

   * Math#multiplyExact(long, long)} instead. Note that if both arguments are {@code int} values,
   * writing {@code Math.multiplyExact(a, b)} will call the {@link Math#multiplyExact(int, int)}
   * overload, not {@link Math#multiplyExact(long, long)}. Also note that multiplying two {@code
   * int} values can <b>never</b> overflow a {@code long}, so you can just write {@code (long) a *
   * b}.
   *
   * @throws ArithmeticException if {@code a * b} overflows in signed {@code long} arithmetic
   */
  @InlineMe(replacement = "Math.multiplyExact(a, b)")
  public static long checkedMultiply(long a, long b) {
    return Math.multiplyExact(a, b);
  }

  /**
   * Returns the {@code b} to the {@code k}th power, provided it does not overflow.
   *
   * @throws ArithmeticException if {@code b} to the {@code k}th power overflows in signed {@code
   *     long} arithmetic
   */
  @GwtIncompatible // TODO
