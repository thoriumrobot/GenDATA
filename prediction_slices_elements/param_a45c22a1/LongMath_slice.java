// Source-based slice around line 641
// Method: <com.google.common.math.LongMath: long saturatedAdd(long,long)>


  /**
   * Returns the sum of {@code a} and {@code b} unless it would overflow or underflow in which case
   * {@code Long.MAX_VALUE} or {@code Long.MIN_VALUE} is returned, respectively.
   *
   * @since 20.0
   */
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static long saturatedAdd(long a, long b) {
    long naiveSum = a + b;
    if ((a ^ b) < 0 | (a ^ naiveSum) >= 0) {
      // If a and b have different signs or a has the same sign as the result then there was no
      // overflow, return.
      return naiveSum;
    }
    // we did over/under flow, if the sign is negative we should return MAX otherwise MIN
    return Long.MAX_VALUE + ((naiveSum >>> (Long.SIZE - 1)) ^ 1);
  }

