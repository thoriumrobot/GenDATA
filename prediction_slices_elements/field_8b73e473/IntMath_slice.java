// Source-based slice around line 202
// Method: com.google.common.math.IntMath.powersOf10


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
  @VisibleForTesting
  static final int[] halfPowersOf10 = {
    3, 31, 316, 3162, 31622, 316227, 3162277, 31622776, 316227766, Integer.MAX_VALUE
  };

  /**
