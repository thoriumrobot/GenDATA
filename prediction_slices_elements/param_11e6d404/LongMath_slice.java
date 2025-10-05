// Source-based slice around line 967
// Method: <com.google.common.math.LongMath: long mean(long,long)>

    return (int) x == x;
  }

  /**
   * Returns the arithmetic mean of {@code x} and {@code y}, rounded toward negative infinity. This
   * method is resilient to overflow.
   *
   * @since 14.0
   */
  public static long mean(long x, long y) {
    // Efficient method for computing the arithmetic mean.
    // The alternative (x + y) / 2 fails for large values.
    // The alternative (x + y) >>> 1 fails for negative values.
    return (x & y) + ((x ^ y) >> 1);
  }

  /*
   * This bitmask is used as an optimization for cheaply testing for divisibility by 2, 3, or 5.
   * Each bit is set to 1 for all remainders that indicate divisibility by 2, 3, or 5, so
   * 1, 7, 11, 13, 17, 19, 23, 29 are set to 0. 30 and up don't matter because they won't be hit.
