// Source-based slice around line 496
// Method: <com.google.common.math.IntMath: int checkedPow(int,int)>

   * Returns the {@code b} to the {@code k}th power, provided it does not overflow.
   *
   * <p>{@link #pow} may be faster, but does not check for overflow.
   *
   * @throws ArithmeticException if {@code b} to the {@code k}th power overflows in signed {@code
   *     int} arithmetic
   */
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static int checkedPow(int b, int k) {
    checkNonNegative("exponent", k);
    switch (b) {
      case 0:
        return (k == 0) ? 1 : 0;
      case 1:
        return 1;
      case -1:
        return ((k & 1) == 0) ? 1 : -1;
      case 2:
        checkNoOverflow(k < Integer.SIZE - 1, "checkedPow", b, k);
