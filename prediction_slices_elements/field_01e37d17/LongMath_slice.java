// Source-based slice around line 200
// Method: com.google.common.math.LongMath.maxLog10ForLeadingZeros

    /*
     * y is the higher of the two possible values of floor(log10(x)). If x < 10^y, then we want the
     * lower of the two possible values, or y - 1, otherwise, we want y.
     */
    return y - lessThanBranchFree(x, powersOf10[y]);
  }

  // maxLog10ForLeadingZeros[i] == floor(log10(2^(Long.SIZE - i)))
  @VisibleForTesting
  static final byte[] maxLog10ForLeadingZeros = {
    19, 18, 18, 18, 18, 17, 17, 17, 16, 16, 16, 15, 15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12,
    12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3,
    3, 2, 2, 2, 1, 1, 1, 0, 0, 0
  };

  @GwtIncompatible // TODO
  @VisibleForTesting
  static final long[] powersOf10 = {
    1L,
    10L,
