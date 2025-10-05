// Source-based slice around line 196
// Method: com.google.common.math.IntMath.maxLog10ForLeadingZeros

    /*
     * y is the higher of the two possible values of floor(log10(x)). If x < 10^y, then we want the
     * lower of the two possible values, or y - 1, otherwise, we want y.
     */
    return y - lessThanBranchFree(x, powersOf10[y]);
  }

  // maxLog10ForLeadingZeros[i] == floor(log10(2^(Long.SIZE - i)))
  @VisibleForTesting
  static final byte[] maxLog10ForLeadingZeros = {
    9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0,
    0
  };

  @VisibleForTesting
  static final int[] powersOf10 = {
    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000
  };

  // halfPowersOf10[i] = largest int less than 10^(i + 0.5)
