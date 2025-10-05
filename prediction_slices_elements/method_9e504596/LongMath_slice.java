// Source-based slice around line 802
// Method: <com.google.common.math.LongMath: long binomial(int,int)>

    1L * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11 * 12 * 13 * 14 * 15 * 16 * 17 * 18 * 19 * 20
  };

  /**
   * Returns {@code n} choose {@code k}, also known as the binomial coefficient of {@code n} and
   * {@code k}, or {@link Long#MAX_VALUE} if the result does not fit in a {@code long}.
   *
   * @throws IllegalArgumentException if {@code n < 0}, {@code k < 0}, or {@code k > n}
   */
  public static long binomial(int n, int k) {
    checkNonNegative("n", n);
    checkNonNegative("k", k);
    checkArgument(k <= n, "k (%s) > n (%s)", k, n);
    if (k > (n >> 1)) {
      k = n - k;
    }
    switch (k) {
      case 0:
        return 1;
      case 1:
