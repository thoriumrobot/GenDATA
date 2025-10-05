// Source-based slice around line 660
// Method: <com.google.common.math.LongMath: long saturatedSubtract(long,long)>


  /**
   * Returns the difference of {@code a} and {@code b} unless it would overflow or underflow in
   * which case {@code Long.MAX_VALUE} or {@code Long.MIN_VALUE} is returned, respectively.
   *
   * @since 20.0
   */
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static long saturatedSubtract(long a, long b) {
    long naiveDifference = a - b;
    if ((a ^ b) >= 0 | (a ^ naiveDifference) >= 0) {
      // If a and b have the same signs or a has the same sign as the result then there was no
      // overflow, return.
      return naiveDifference;
    }
    // we did over/under flow
    return Long.MAX_VALUE + ((naiveDifference >>> (Long.SIZE - 1)) ^ 1);
  }

