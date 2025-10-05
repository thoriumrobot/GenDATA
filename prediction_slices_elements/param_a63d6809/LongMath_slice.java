// Source-based slice around line 679
// Method: <com.google.common.math.LongMath: long saturatedMultiply(long,long)>


  /**
   * Returns the product of {@code a} and {@code b} unless it would overflow or underflow in which
   * case {@code Long.MAX_VALUE} or {@code Long.MIN_VALUE} is returned, respectively.
   *
   * @since 20.0
   */
  // Whenever both tests are cheap and functional, it's faster to use &, | instead of &&, ||
  @SuppressWarnings("ShortCircuitBoolean")
  public static long saturatedMultiply(long a, long b) {
    // see checkedMultiply for explanation
    int leadingZeros =
        Long.numberOfLeadingZeros(a)
            + Long.numberOfLeadingZeros(~a)
            + Long.numberOfLeadingZeros(b)
            + Long.numberOfLeadingZeros(~b);
    if (leadingZeros > Long.SIZE + 1) {
      return a * b;
    }
    // the return value if we will overflow (which we calculate by overflowing a long :) )
