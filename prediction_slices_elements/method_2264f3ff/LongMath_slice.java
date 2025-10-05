// Source-based slice around line 562
// Method: <com.google.common.math.LongMath: long checkedSubtract(long,long)>

   * Math#subtractExact(long, long)} instead. Note that if both arguments are {@code int} values,
   * writing {@code Math.subtractExact(a, b)} will call the {@link Math#subtractExact(int, int)}
   * overload, not {@link Math#subtractExact(long, long)}. Also note that subtracting two {@code
   * int} values can <b>never</b> overflow a {@code long}, so you can just write {@code (long) a -
   * b}.
   *
   * @throws ArithmeticException if {@code a - b} overflows in signed {@code long} arithmetic
   */
  @InlineMe(replacement = "Math.subtractExact(a, b)")
  public static long checkedSubtract(long a, long b) {
    return Math.subtractExact(a, b);
  }

  /**
   * Returns the product of {@code a} and {@code b}, provided it does not overflow.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use {@link
   * Math#multiplyExact(long, long)} instead. Note that if both arguments are {@code int} values,
   * writing {@code Math.multiplyExact(a, b)} will call the {@link Math#multiplyExact(int, int)}
   * overload, not {@link Math#multiplyExact(long, long)}. Also note that multiplying two {@code
