// Source-based slice around line 456
// Method: <com.google.common.math.IntMath: int checkedAdd(int,int)>

  /**
   * Returns the sum of {@code a} and {@code b}, provided it does not overflow.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use {@link
   * Math#addExact(int, int)} instead.
   *
   * @throws ArithmeticException if {@code a + b} overflows in signed {@code int} arithmetic
   */
  @InlineMe(replacement = "Math.addExact(a, b)")
  public static int checkedAdd(int a, int b) {
    return Math.addExact(a, b);
  }

  /**
   * Returns the difference of {@code a} and {@code b}, provided it does not overflow.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use {@link
   * Math#subtractExact(int, int)} instead.
   *
   * @throws ArithmeticException if {@code a - b} overflows in signed {@code int} arithmetic
