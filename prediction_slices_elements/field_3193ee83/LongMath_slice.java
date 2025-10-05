// Source-based slice around line 233
// Method: com.google.common.math.LongMath.halfPowersOf10

    1000000000000000L,
    10000000000000000L,
    100000000000000000L,
    1000000000000000000L
  };

  // halfPowersOf10[i] = largest long less than 10^(i + 0.5)
  @GwtIncompatible // TODO
  @VisibleForTesting
  static final long[] halfPowersOf10 = {
    3L,
    31L,
    316L,
    3162L,
    31622L,
    316227L,
    3162277L,
    31622776L,
    316227766L,
    3162277660L,
