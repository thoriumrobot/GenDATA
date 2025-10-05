// Source-based slice around line 653
// Method: <com.google.common.math.IntMath: int binomial(int,int)>

    1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12
  };

  /**
   * Returns {@code n} choose {@code k}, also known as the binomial coefficient of {@code n} and
   * {@code k}, or {@link Integer#MAX_VALUE} if the result does not fit in an {@code int}.
   *
   * @throws IllegalArgumentException if {@code n < 0}, {@code k < 0} or {@code k > n}
   */
  public static int binomial(int n, int k) {
    checkNonNegative("n", n);
    checkNonNegative("k", k);
    checkArgument(k <= n, "k (%s) > n (%s)", k, n);
    if (k > (n >> 1)) {
      k = n - k;
    }
    if (k >= biggestBinomials.length || n > biggestBinomials[k]) {
      return Integer.MAX_VALUE;
    }
    switch (k) {
