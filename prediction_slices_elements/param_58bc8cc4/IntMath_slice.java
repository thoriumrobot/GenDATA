// Source-based slice around line 706
// Method: <com.google.common.math.IntMath: int mean(int,int)>

    33
  };

  /**
   * Returns the arithmetic mean of {@code x} and {@code y}, rounded towards negative infinity. This
   * method is overflow resilient.
   *
   * @since 14.0
   */
  public static int mean(int x, int y) {
    // Efficient method for computing the arithmetic mean.
    // The alternative (x + y) / 2 fails for large values.
    // The alternative (x + y) >>> 1 fails for negative values.
    return (x & y) + ((x ^ y) >> 1);
  }

  /**
   * Returns {@code true} if {@code n} is a <a
   * href="http://mathworld.wolfram.com/PrimeNumber.html">prime number</a>: an integer <i>greater
   * than one</i> that cannot be factored into a product of <i>smaller</i> positive integers.
