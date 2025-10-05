// Source-based slice around line 572
// Method: <com.google.common.math.IntMath: int saturatedPow(int,int)>


  /**
   * Returns the {@code b} to the {@code k}th power, unless it would overflow or underflow in which
   * case {@code Integer.MAX_VALUE} or {@code Integer.MIN_VALUE} is returned, respectively.
   *
   * @since 20.0
   */
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static int saturatedPow(int b, int k) {
    checkNonNegative("exponent", k);
    switch (b) {
      case 0:
        return (k == 0) ? 1 : 0;
      case 1:
        return 1;
      case -1:
        return ((k & 1) == 0) ? 1 : -1;
      case 2:
        if (k >= Integer.SIZE - 1) {
