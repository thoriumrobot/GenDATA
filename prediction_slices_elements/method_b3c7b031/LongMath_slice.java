// Source-based slice around line 592
// Method: <com.google.common.math.LongMath: long checkedPow(long,int)>

  /**
   * Returns the {@code b} to the {@code k}th power, provided it does not overflow.
   *
   * @throws ArithmeticException if {@code b} to the {@code k}th power overflows in signed {@code
   *     long} arithmetic
   */
  @GwtIncompatible // TODO
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static long checkedPow(long b, int k) {
    checkNonNegative("exponent", k);
    if (b >= -2 & b <= 2) {
      switch ((int) b) {
        case 0:
          return (k == 0) ? 1 : 0;
        case 1:
          return 1;
        case -1:
          return ((k & 1) == 0) ? 1 : -1;
        case 2:
