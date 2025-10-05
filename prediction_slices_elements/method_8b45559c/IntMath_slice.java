// Source-based slice around line 482
// Method: <com.google.common.math.IntMath: int checkedMultiply(int,int)>

  /**
   * Returns the product of {@code a} and {@code b}, provided it does not overflow.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use {@link
   * Math#multiplyExact(int, int)} instead.
   *
   * @throws ArithmeticException if {@code a * b} overflows in signed {@code int} arithmetic
   */
  @InlineMe(replacement = "Math.multiplyExact(a, b)")
  public static int checkedMultiply(int a, int b) {
    return Math.multiplyExact(a, b);
  }

  /**
   * Returns the {@code b} to the {@code k}th power, provided it does not overflow.
   *
   * <p>{@link #pow} may be faster, but does not check for overflow.
   *
   * @throws ArithmeticException if {@code b} to the {@code k}th power overflows in signed {@code
   *     int} arithmetic
