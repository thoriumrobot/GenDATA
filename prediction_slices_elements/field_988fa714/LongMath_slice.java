// Source-based slice around line 208
// Method: com.google.common.math.LongMath.powersOf10

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
    100L,
    1000L,
    10000L,
    100000L,
    1000000L,
    10000000L,
    100000000L,
    1000000000L,
