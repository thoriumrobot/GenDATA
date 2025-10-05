// Source-based slice around line 979
// Method: com.google.common.math.LongMath.SIEVE_30

    // The alternative (x + y) >>> 1 fails for negative values.
    return (x & y) + ((x ^ y) >> 1);
  }

  /*
   * This bitmask is used as an optimization for cheaply testing for divisibility by 2, 3, or 5.
   * Each bit is set to 1 for all remainders that indicate divisibility by 2, 3, or 5, so
   * 1, 7, 11, 13, 17, 19, 23, 29 are set to 0. 30 and up don't matter because they won't be hit.
   */
  private static final int SIEVE_30 =
      ~((1 << 1) | (1 << 7) | (1 << 11) | (1 << 13) | (1 << 17) | (1 << 19) | (1 << 23)
          | (1 << 29));

  /**
   * Returns {@code true} if {@code n} is a <a
   * href="http://mathworld.wolfram.com/PrimeNumber.html">prime number</a>: an integer <i>greater
   * than one</i> that cannot be factored into a product of <i>smaller</i> positive integers.
   * Returns {@code false} if {@code n} is zero, one, or a composite number (one which <i>can</i> be
   * factored into smaller positive integers).
   *
