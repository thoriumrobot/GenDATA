// Source-based slice around line 469
// Method: <com.google.common.math.IntMath: int checkedSubtract(int,int)>

  /**
   * Returns the difference of {@code a} and {@code b}, provided it does not overflow.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use {@link
   * Math#subtractExact(int, int)} instead.
   *
   * @throws ArithmeticException if {@code a - b} overflows in signed {@code int} arithmetic
   */
  @InlineMe(replacement = "Math.subtractExact(a, b)")
  public static int checkedSubtract(int a, int b) {
    return Math.subtractExact(a, b);
  }

  /**
   * Returns the product of {@code a} and {@code b}, provided it does not overflow.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use {@link
   * Math#multiplyExact(int, int)} instead.
   *
   * @throws ArithmeticException if {@code a * b} overflows in signed {@code int} arithmetic
