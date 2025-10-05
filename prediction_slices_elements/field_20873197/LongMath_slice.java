// Source-based slice around line 920
// Method: com.google.common.math.LongMath.biggestSimpleBinomials

    66,
    66
  };

  /*
   * binomial(biggestSimpleBinomials[k], k) doesn't need to use the slower GCD-based impl, but
   * binomial(biggestSimpleBinomials[k] + 1, k) does.
   */
  @VisibleForTesting
  static final int[] biggestSimpleBinomials = {
    Integer.MAX_VALUE,
    Integer.MAX_VALUE,
    Integer.MAX_VALUE,
    2642246,
    86251,
    11724,
    3218,
    1313,
    684,
    419,
