// Source-based slice around line 465
// Method: <com.google.common.math.BigIntegerMath: BigInteger binomial(int,int)>


  /**
   * Returns {@code n} choose {@code k}, also known as the binomial coefficient of {@code n} and
   * {@code k}, that is, {@code n! / (k! (n - k)!)}.
   *
   * <p><b>Warning:</b> the result can take as much as <i>O(k log n)</i> space.
   *
   * @throws IllegalArgumentException if {@code n < 0}, {@code k < 0}, or {@code k > n}
   */
  public static BigInteger binomial(int n, int k) {
    checkNonNegative("n", n);
    checkNonNegative("k", k);
    checkArgument(k <= n, "k (%s) > n (%s)", k, n);
    if (k > (n >> 1)) {
      k = n - k;
    }
    if (k < LongMath.biggestBinomials.length && n <= LongMath.biggestBinomials[k]) {
      return BigInteger.valueOf(LongMath.binomial(n, k));
    }

