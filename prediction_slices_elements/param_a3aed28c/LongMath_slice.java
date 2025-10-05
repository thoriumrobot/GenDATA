// Source-based slice around line 710
// Method: <com.google.common.math.LongMath: long saturatedPow(long,int)>


  /**
   * Returns the {@code b} to the {@code k}th power, unless it would overflow or underflow in which
   * case {@code Long.MAX_VALUE} or {@code Long.MIN_VALUE} is returned, respectively.
   *
   * @since 20.0
   */
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static long saturatedPow(long b, int k) {
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
