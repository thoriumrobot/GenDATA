// Source-based slice around line 545
// Method: <com.google.common.math.LongMath: long checkedAdd(long,long)>

   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use {@link
   * Math#addExact(long, long)} instead. Note that if both arguments are {@code int} values, writing
   * {@code Math.addExact(a, b)} will call the {@link Math#addExact(int, int)} overload, not {@link
   * Math#addExact(long, long)}. Also note that adding two {@code int} values can <b>never</b>
   * overflow a {@code long}, so you can just write {@code (long) a + b}.
   *
   * @throws ArithmeticException if {@code a + b} overflows in signed {@code long} arithmetic
   */
  @InlineMe(replacement = "Math.addExact(a, b)")
  public static long checkedAdd(long a, long b) {
    return Math.addExact(a, b);
  }

  /**
   * Returns the difference of {@code a} and {@code b}, provided it does not overflow.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use {@link
   * Math#subtractExact(long, long)} instead. Note that if both arguments are {@code int} values,
   * writing {@code Math.subtractExact(a, b)} will call the {@link Math#subtractExact(int, int)}
   * overload, not {@link Math#subtractExact(long, long)}. Also note that subtracting two {@code
