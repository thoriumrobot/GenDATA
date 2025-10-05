// Source-based slice around line 208
// Method: com.google.common.math.IntMath.halfPowersOf10

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
   * Returns {@code b} to the {@code k}th power. Even if the result overflows, it will be equal to
   * {@code BigInteger.valueOf(b).pow(k).intValue()}. This implementation runs in {@code O(log k)}
   * time.
   *
   * <p>Compare {@link #checkedPow}, which throws an {@link ArithmeticException} upon overflow.
   *
