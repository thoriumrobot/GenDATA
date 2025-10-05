// Source-based slice around line 878
// Method: com.google.common.math.LongMath.biggestBinomials

    denominator /= commonDivisor;
    // We know gcd(x, denominator) = 1, and x * numerator / denominator is exact,
    // so denominator must be a divisor of numerator.
    return x * (numerator / denominator);
  }

  /*
   * binomial(biggestBinomials[k], k) fits in a long, but not binomial(biggestBinomials[k] + 1, k).
   */
  static final int[] biggestBinomials = {
    Integer.MAX_VALUE,
    Integer.MAX_VALUE,
    Integer.MAX_VALUE,
    3810779,
    121977,
    16175,
    4337,
    1733,
    887,
    534,
